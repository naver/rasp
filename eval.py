import argparser
import os
from utils.logger import Logger
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import get_dataset, get_fgr_dataset
from metrics import StreamSegMetrics
from train import Trainer
from utils.utils import visualize_images, save_images, _init_dist_slurm, visualize_external
from tasks import get_per_task_classes


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
        #device_id, device = int(os.environ['LOCAL_RANK']), torch.device(int(os.environ['LOCAL_RANK']))
        device_id, device = opts.local_rank, torch.device(opts.local_rank)
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

    # Initialize logging
    task_name = f"{opts.dataset}-{opts.task}"
    if opts.overlap and opts.dataset == 'voc':
        task_name += "-ov"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"


    #logger = WandBLogger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step,
    #                     name=f"{task_name}_{opts.name}")

    logger = Logger(logdir_full, rank=rank, type='torch', debug=opts.debug, filename=os.path.join(logdir_full, f'eval-log-step-{opts.step}.txt'),
                    summary=opts.visualize, step=opts.step, name=f"{task_name}_{opts.name}")

    ckpt_path = f"{opts.ckpt_root}/step/{task_name}/{opts.name}_{opts.step}.pth"

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

    if opts.dataset == 'cub':
        train_dst, val_dst, test_dst, labels, n_classes = get_fgr_dataset(opts)
    else:
        if opts.external_dataset:
            train_dst, val_dst, test_dst, labels, n_classes, ext_dst = get_dataset(opts)
        else:
            train_dst, val_dst, test_dst, labels, n_classes = get_dataset(opts)
    
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1, shuffle=False,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    if opts.external_dataset:
        ext_loader = data.DataLoader(ext_dst, batch_size=opts.batch_size,
                                     sampler=DistributedSampler(ext_dst, num_replicas=world_size, rank=rank),
                                     num_workers=opts.num_workers, drop_last=True)
    else:
        ext_loader = None

    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    if opts.external_dataset:
        logger.info(f"External train set: {len(ext_dst)}")
    
    logger.info(f"Total batch size is {opts.batch_size * world_size}")
    opts.max_iters = opts.epochs * len(train_loader)
    if opts.lr_policy == "warmup":
        opts.start_decay = opts.pseudo_ep * len(train_loader)

    # dump the argparse arguments
    argparse_str = '\n'.join(f'{k}={v}' for k, v in vars(opts).items()) + '\n'
    logger.info(argparse_str)

    # xxx Set up Trainer
    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(logger, device=device, opts=opts)

    # xxx Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None and opts.curr_step_ckpt is not None:
            prev_path = opts.step_ckpt
            curr_path = opts.curr_step_ckpt
            
            trainer.load_ckpt_inference(prev_path, curr_path)
        else:
            raise ValueError(f'Either of current or previous checkpoints are missing. Please provide both!')
    else:
        raise ValueError(f'Inference and visualization at step 0 is not supported yet!')

    # print opts
    logger.add_config(opts)

    val_metrics = StreamSegMetrics(
                        n_classes=n_classes,
                        n_sup_classes=get_per_task_classes(
                            dataset=opts.dataset,
                            name=opts.task,
                            step=opts.step)[0]
    )

    affinity_matrix = train_dst.class_affinity(scaling=opts.tau, similarity_type=opts.similarity_type) if opts.semantic_similarity else None

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    torch.distributed.barrier()
    
    # xxx From here starts the test code
    logger.info(f"*** Test the model on all seen classes at step t={opts.step}")

    # make data loader
    # batch size must be set to 1, otherwise the visualization pipeline will break

    if opts.dataset == 'coco-voc':
        # additionally evaluate on the VOC validation set
        voc_val_metrics = StreamSegMetrics(
                                n_classes=n_classes,
                                n_sup_classes=get_per_task_classes(
                                dataset=opts.dataset,
                                name=opts.task,
                                step=opts.step)[0]
                            )
        save_folder = os.path.join(logdir_full, 'viz/voc/', 'step-'+str(opts.step))
        if not os.path.exists(save_folder) and rank == 0:
            os.makedirs(save_folder)
        
        val_score, _ = trainer.validate_and_visualize(
                            loader=val_loader, 
                            metrics=voc_val_metrics,
                            affinity_matrix=affinity_matrix,
                            class_dict=train_dst.class_dict(),
                            save_folder=save_folder,
                            visualize=opts.visualize_images
                        )
        
        logger.info(f"FINAL SCORES for VOC after step t={opts.step}")
        logger.info(val_metrics.to_str(val_score))

    # evaluate on the validation set
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    save_folder = os.path.join(logdir_full, f"viz/{opts.dataset.split('-')[0]}/", 'step-'+str(opts.step))
    if not os.path.exists(save_folder) and rank == 0:
        os.makedirs(save_folder)

    val_score, _ = trainer.validate_and_visualize(
                            loader=test_loader, 
                            metrics=val_metrics,
                            affinity_matrix=affinity_matrix,
                            class_dict=train_dst.class_dict(),
                            save_folder=save_folder,
                            visualize=opts.visualize_images
                    )

    logger.info(f"*** End of Test")
    logger.info(f"FINAL SCORES for {opts.dataset.split('-')[0]} after step t={opts.step}")
    logger.info(val_metrics.to_str(val_score))
    logger.info(f"(All metric that includes the background class): Mean Acc: {val_score['Mean Acc']}; Mean IoU: {val_score['Mean IoU']}")
    logger.commit()
    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs(f"{opts.ckpt_root}/step", exist_ok=True)
    main(opts)
