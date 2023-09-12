import argparser
import os
from utils.logger import WandBLogger, Logger
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import get_dataset, get_fss_dataset
from metrics import StreamSegMetricsFSS
from train import Trainer
from utils.utils import visualize_images, save_images, _init_dist_slurm, visualize_external
from tasks import Task
import time

def save_ckpt(path, trainer, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_state": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "best_score": best_score,
    }
    if trainer.pseudolabeler is not None:
        state["pseudolabeler"] = trainer.pseudolabeler.state_dict()

    torch.save(state, path)


def main(opts):
    if opts.launcher == 'pytorch':
        distributed.init_process_group(backend='nccl', init_method='env://')
        device_id, device = int(os.environ['LOCAL_RANK']), torch.device(int(os.environ['LOCAL_RANK']))
        #device_id, device = opts.local_rank, torch.device(opts.local_rank)
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
        torch.cuda.set_device(device_id)
        opts.device_id = device_id
    elif opts.launcher == 'slurm':
        device_id, world_size = _init_dist_slurm(backend='nccl')
        opts.device_id = device_id
        rank = device_id
        device = torch.device(rank)
    else:
        raise ValueError("Unsupported launcher type.")

    task = Task(opts)

    # Initialize logging
    task_name = f"{opts.dataset}-{opts.task}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    
    if opts.overlap and opts.dataset == 'voc':
        task_name += "-ov"
    
    if task.nshot != -1:
        logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/s{task.nshot}-i{task.ishot}"
    else:
        logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"


    #logger = WandBLogger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step,
    #                     name=f"{task_name}_{opts.name}")

    logger = Logger(logdir_full, rank=rank, type='torch', debug=opts.debug, filename=os.path.join(logdir_full, 'log.txt'),
                    summary=opts.visualize, step=opts.step, name=f"{task_name}_{opts.name}")

    ckpt_path = f"{opts.ckpt_root}/step/{task_name}/{name}_{opts.step}.pth"

    if not os.path.exists(f"{opts.ckpt_root}/step/{task_name}") and rank == 0:
        os.makedirs(f"{opts.ckpt_root}/step/{task_name}")

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    opts.batch_size = opts.batch_size // world_size
    #train_dst, val_dst, test_dst, labels, n_classes = get_dataset(opts)
    train_dst, val_dst, test_dst = get_fss_dataset(opts, task)
    
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst, batch_size=min(opts.batch_size, len(train_dst)),
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=min(opts.batch_size, len(val_dst)), shuffle=False,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    if opts.external_dataset:
        ext_loader = data.DataLoader(ext_dst, batch_size=opts.batch_size,
                                     sampler=DistributedSampler(ext_dst, num_replicas=world_size, rank=rank),
                                     num_workers=opts.num_workers, drop_last=True)
    else:
        ext_loader = None

    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {task.num_classes}")
    train_iterations = 1 if task.step == 0 else 20 // task.nshot
    if opts.iter is not None:
        opts.epochs = opts.iter // (len(train_loader) * train_iterations)
    logger.info(f'Total batch size is {min(opts.batch_size, len(train_dst)) * world_size}')
    logger.info(f'The train loader contains {len(train_loader)} iterations per epoch, multiplied by {train_iterations}. Total epochs are {opts.epochs}')

    if opts.external_dataset:
        logger.info(f"External train set: {len(ext_dst)}")
    opts.max_iters = opts.epochs * len(train_loader)
    if opts.lr_policy == "warmup":
        opts.start_decay = opts.pseudo_ep * len(train_loader)

    # xxx Set up Trainer
    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(logger, device=device, opts=opts, task=task)

    # xxx Load old model from old weights if step > 0!
    if task.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"{opts.ckpt_root}/step/{task_name}/{opts.name}_{opts.step - 1}.pth"
        trainer.load_step_ckpt(path)

    # Load training checkpoint if any
    if opts.continue_ckpt:
        opts.ckpt = ckpt_path
    if opts.ckpt is not None:
        cur_epoch, best_score = trainer.load_ckpt(opts.ckpt)
    else:
        logger.info("[!] Start from epoch 0")
        cur_epoch = 0
        best_score = 0.

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_config(opts)

    TRAIN = not opts.test
    train_metrics = StreamSegMetricsFSS(len(task.get_order()), task.get_n_classes()[0])
    val_metrics = StreamSegMetricsFSS(len(task.get_order()), task.get_n_classes()[0])
    results = {}

    affinity_matrix = train_dst.class_affinity(scaling=opts.tau, similarity_type=opts.similarity_type) if opts.semantic_similarity else None

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here
    # ===== Visualize old model predictions before training on incremental steps =====
    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        start = time.time()
        epoch_loss = trainer.train(cur_epoch=cur_epoch, 
                                   train_loader=train_loader, 
                                   external_loader=ext_loader, 
                                   affinity_matrix=affinity_matrix,
                                   metrics=train_metrics)
        end = time.time()
        train_score = train_metrics.get_results()
        len_ep = int(end - start)
        
        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]} \n"
                    f"Train_Acc={train_score['Overall Acc']:.4f}, Train_Iou={train_score['Mean IoU']:.4f}"
                    f"\n -- time: {len_ep // 60}:{len_ep % 60} -- ")
        logger.info(f"I will finish in {len_ep * (opts.epochs - cur_epoch) // 60} minutes")

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            val_score, ret_val_samples = trainer.validate(loader=val_loader, metrics=val_metrics, 
                                                          affinity_matrix=affinity_matrix)

            logger.print("Done validation Model")

            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                # best model to build incremental steps
                save_ckpt(ckpt_path, trainer, cur_epoch, score)
                logger.info("[!] Checkpoint saved.")

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

            logger.commit()
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}")

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(ckpt_path, trainer, cur_epoch, best_score)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.test_batch_size,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)
    val_score, ret_test_samples = trainer.validate(loader=test_loader, metrics=val_metrics,
                                                   affinity_matrix=affinity_matrix)
    logger.info(f"*** End of Test")
    logger.info(val_metrics.to_str(val_score))
    ret = val_score['Mean IoU']

    logger.log_results(task=task, name=opts.name, results=val_score['Class IoU'].values())
    logger.log_aggregates(task=task, name=opts.name, results=val_score['Agg'])

    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_fss_command_options(opts)

    os.makedirs(f"{opts.ckpt_root}/step", exist_ok=True)
    main(opts)
