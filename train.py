import torch
import numpy as np
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import tqdm
from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss
from torch.cuda import amp
from segmentation_module import make_model, TestAugmentation
import tasks
from torch.nn.parallel import DistributedDataParallel
import os.path as osp
from wss.modules import PAMR, ASPP
from utils.utils import denorm, label_to_one_hot
from wss.single_stage import pseudo_gtmask, balanced_mask_loss_ce, balanced_mask_loss_unce
from utils.wss_loss import bce_loss, ngwp_focal, binarize, sem_bce_loss
from segmentation_module import get_norm
from utils.scheduler import get_scheduler
import pdb
from torchvision.utils import make_grid, save_image
import os
from utils.utils import denorm, visualize_predictions

class Trainer:
    def __init__(self, logger, device, opts, task=None):
        self.logger = logger
        self.device = device
        self.opts = opts
        self.scaler = amp.GradScaler()

        self.sample_num = opts.sample_num
        self.pl_threshold = opts.pl_threshold
        self.ws_bkg = opts.ws_bkg
        self.viz_dataset = opts.dataset
        self.external_dataset = opts.external_dataset
        self.semantic_similarity = opts.semantic_similarity
        self.lambda_sem = opts.lambda_sem

        if task is not None:
            # in case of FSS
            self.classes = classes = task.get_n_classes()
            self.order = task.get_order()
        else:
            self.classes = classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
            self.order = tasks.get_order(opts.dataset, opts.task, opts.step)

        if classes is not None:
            new_classes = classes[-1]
            self.tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = self.tot_classes - new_classes
        else:
            self.old_classes = 0

        self.model = make_model(opts, classes=classes)

        if opts.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            if task is not None:
                # in case of FSS
                prev_classes = task.get_n_classes()[:-1]
            else:
                prev_classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1)

            self.model_old = make_model(opts, classes=prev_classes)
            self.model_old.to(self.device)
            # freeze old model and set eval mode
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        self.weakly = opts.weakly and opts.step > 0
        self.pos_w = opts.pos_w
        self.use_aff = opts.affinity
        self.weak_single_stage_dist = opts.ss_dist
        self.pseudo_epoch = opts.pseudo_ep
        cls_classes = self.tot_classes
        self.pseudolabeler = None

        if self.weakly:
             # PAMR module is not trainable (no backprop occurs)
            self.affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12]).to(device)
            for p in self.affinity.parameters():
                p.requires_grad = False

            # initialize the localizer (or auxiliary classifier). This branch learns the seg masks from image labels
            norm = get_norm(opts)
            channels = 4096 if "wide" in opts.backbone else 2048
            self.pseudolabeler = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               norm(256),
                                               nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               norm(256),
                                               nn.Conv2d(256, cls_classes, kernel_size=1, stride=1))

            self.icarl = opts.icarl

        self.optimizer, self.scheduler = self.get_optimizer(opts)

        # send models to DDP
        self.distribute(opts)

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de # Distillation on Encoder features
        self.lde_flag = self.lde > 0. and self.model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and self.model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and self.model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and self.model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

    def get_optimizer(self, opts):
        params = []
        if not opts.freeze:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                           'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
        if self.weakly:
            params.append({"params": filter(lambda p: p.requires_grad, self.pseudolabeler.parameters()),
                           'weight_decay': opts.weight_decay, 'lr': opts.lr_pseudo})

        optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)
        scheduler = get_scheduler(opts, optimizer)

        return optimizer, scheduler

    def distribute(self, opts):
        self.model = DistributedDataParallel(self.model.to(self.device), device_ids=[opts.device_id],
                                             output_device=opts.device_id, find_unused_parameters=False)
        if self.weakly:
            self.pseudolabeler = DistributedDataParallel(self.pseudolabeler.to(self.device), device_ids=[opts.device_id],
                                                         output_device=opts.device_id, find_unused_parameters=False)

    def train(self, cur_epoch, train_loader, external_loader=None, print_int=10, affinity_matrix=None, metrics=None):
        """Train and return epoch loss"""
        if metrics is not None:
            metrics.reset()
        optim = self.optimizer
        scheduler = self.scheduler
        device = self.device
        model = self.model
        criterion = self.criterion
        logger = self.logger

        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        rank_zero = distributed.get_rank() == 0 # a boolean variable, true if the rank of the process is 0

        epoch_loss = 0.0
        reg_loss = 0.0
        l_cam_out = 0.0
        l_cam_int = 0.0
        l_cam_new = 0.0
        l_sem_sim = 0.0
        l_seg = 0.0
        l_cls = 0.0
        l_loc = 0.0
        l_ext_loc = 0.0
        l_seg_ext = 0.0
        interval_loss = 0.0

        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        lde_ext = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        train_loader.sampler.set_epoch(cur_epoch)

        if distributed.get_rank() == 0:
            tq = tqdm.tqdm(total=len(train_loader))
            tq.set_description("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        else:
            tq = None
        
        if self.external_dataset:
            ext_iter = iter(external_loader)

        model.train()
        for cur_step, (images, labels, l1h) in enumerate(train_loader):

            # load external data set images
            if self.weakly and self.external_dataset:
                try:
                    ext_data = ext_iter.next()
                    ext_images, ext_l1h = ext_data[0], ext_data[1]
                except:
                    ext_iter = iter(external_loader)
                    ext_data = ext_iter.next()
                    ext_images, ext_l1h = ext_data[0], ext_data[1]

                ext_images = ext_images.to(device, dtype=torch.float)
                ext_l1h = ext_l1h.to(device, dtype=torch.float)

            images = images.to(device, dtype=torch.float)   # B x 3 x H x W
            # index 0 in l1h represents the first object/thing class
            l1h = l1h.to(device, dtype=torch.float)         # these are one_hot, i.e., B x nb_classes
            labels = labels.to(device, dtype=torch.long)    # B x H x W

            if self.weakly and self.external_dataset:
                images = torch.cat([images, ext_images], dim=0)
                l1h = torch.cat([l1h, ext_l1h], dim=0)

            # an indicator which is 1 for images belonging to new classes
            # and 0 for images belonging to old classes
            new_classes_mask = (l1h.sum(-1) > 0).float()

            with amp.autocast():
                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.weakly) and self.model_old is not None:
                    with torch.no_grad():
                        # outputs_old has shape bs x (nb_old_classes + bg) x h x w
                        outputs_old, features_old = self.model_old(images, interpolate=False)

                optim.zero_grad()
                outputs, features = model(images, interpolate=False) #outputs shape: bs x nb_classes x h x w

                # xxx BCE / Cross Entropy Loss
                if not self.weakly:
                    outputs = F.interpolate(outputs, size=images.shape[-2:], mode="bilinear", align_corners=False) # shape B x nb_classes x H x W. upsampled.
                    if not self.icarl_only_dist:
                        loss = criterion(outputs, labels)  # B x H x W
                    else:
                        # ICaRL loss -- unique CE+KD
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                    loss = loss.mean()  # scalar

                    # xxx ICARL DISTILLATION
                    if self.icarl_combined:
                        # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                        n_cl_old = outputs_old.shape[1]
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                        l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                      torch.sigmoid(outputs_old))

                    # xxx ILTSS (distillation on features or logits)
                    if self.lde_flag:
                        lde = self.lde * self.lde_loss(features['body'], features_old['body'])

                    if self.lkd_flag:
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        # resize new output to remove new logits and keep only the old ones
                        lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

                else:
                    bs = images.shape[0]

                    self.pseudolabeler.eval()
                    int_masks = self.pseudolabeler(features['body']).detach() # shape B x (nb_classes + bg) x 32 x 32, 1-bkg, 15-old, 5-new

                    self.pseudolabeler.train()
                    int_masks_raw = self.pseudolabeler(features['body']) # shape: bs x (nb_classes + bg) x 32 x 32

                    # CAM Loss
                    if self.opts.no_mask:
                        l_cam_new = bce_loss(
                            int_masks_raw, 
                            outputs_old.detach(), 
                            nb_new_classes=self.tot_classes - self.old_classes,
                            labels=l1h,
                            mode=self.opts.cam, 
                            reduction='mean'
                        )
                    else:
                        l_cam_new = bce_loss(
                            int_masks_raw, 
                            outputs_old.detach(), 
                            self.tot_classes - self.old_classes,
                            l1h[:, self.old_classes - 1:],
                            mode=self.opts.cam, 
                            reduction='none',
                            affinity_matrix=affinity_matrix
                        ).sum(-1)
                        l_cam_new = (l_cam_new * new_classes_mask).sum() / (new_classes_mask.sum() + 1e-5)

                    # Semantic Similarity Loss
                    if self.semantic_similarity:
                        l_sem_sim, similarity_weights = sem_bce_loss(
                            int_masks_raw, 
                            labels=l1h, 
                            outputs_old=outputs_old,
                            nb_new_classes=self.tot_classes - self.old_classes,
                            affinity_matrix=affinity_matrix,
                            order=self.order
                        )

                    # Prior Loss
                    l_loc = F.binary_cross_entropy_with_logits(
                        int_masks_raw[:, :self.old_classes],
                        torch.sigmoid(outputs_old.detach()),
                        reduction='mean'
                    )
                    
                    l_sem_sim = self.lambda_sem * l_sem_sim
                    l_cam_int = l_cam_new + l_loc + l_sem_sim
                    
                    # Distillation Loss on the Encoder features.
                    if self.lde_flag:
                        lde = self.lde * self.lde_loss(features['body'], features_old['body'])

                    l_cam_out = 0 * outputs[0, 0].mean()  # avoid errors due to DDP

                    if cur_epoch >= self.pseudo_epoch:
                        
                        # predictions from the pseudo-labeller (or the localizer or auxiliary classifier)
                        int_masks_orig = int_masks.softmax(dim=1)
                        int_masks_soft = int_masks.softmax(dim=1)

                        # use affinity on CAM. this is the PAMR module in Stefan Roth's paper
                        # Single-Stage Semantic Segmentation from Image Labels, CVPR 2020
                        if self.use_aff:
                            image_raw = denorm(images)
                            # downsample the images to the mask resolution
                            im = F.interpolate(image_raw, int_masks.shape[-2:], mode="bilinear",
                                               align_corners=True)
                            int_masks_soft = self.affinity(im, int_masks_soft.detach())

                        int_masks_orig[:, 1:] *= l1h[:, :, None, None] # l1h no bg
                        int_masks_soft[:, 1:] *= l1h[:, :, None, None]

                        # Convert continuous mask into binary mask
                        pseudo_gt_seg = pseudo_gtmask(int_masks_soft, ambiguous=True, cutoff_top=0.6,
                                                      cutoff_bkg=0.7, cutoff_low=0.2).detach()  # B x C x H x W

                        # smoothed pseudo-label for each pixel
                        pseudo_gt_seg_lx = binarize(int_masks_orig) # Hard pseudo-label, shape bs x (nb_classes + bkg) x h x w
                        pseudo_gt_seg_lx = (self.opts.alpha * pseudo_gt_seg_lx) + \
                                           ((1-self.opts.alpha) * int_masks_orig)

                        # ignore_mask = (pseudo_gt_seg.sum(1) > 0)
                        px_cls_per_image = pseudo_gt_seg_lx.view(bs, self.tot_classes, -1).sum(dim=-1)
                        batch_weight = torch.eq((px_cls_per_image[:, self.old_classes:] > 0),
                                                l1h[:, self.old_classes - 1:].bool()) # shape: bs x nb_new_classes

                        batch_weight = (
                                    batch_weight.sum(dim=1) == (self.tot_classes - self.old_classes)).float()

                        # predictions from the old model for the current task images
                        target_old = torch.sigmoid(outputs_old.detach())

                        # combine the PL from the old and current tasks
                        target = torch.cat((target_old, pseudo_gt_seg_lx[:, self.old_classes:]), dim=1)
                        if self.opts.icarl_bkg == -1:
                            # torch.min is the original code
                            target[:, 0] = torch.min(target[:, 0], pseudo_gt_seg_lx[:, 0])
                        else:
                            target[:, 0] = (1-self.opts.icarl_bkg) * target[:, 0] + \
                                           self.opts.icarl_bkg * pseudo_gt_seg_lx[:, 0]

                        # To train the main segmentation head across all the classes at current step
                        l_seg = F.binary_cross_entropy_with_logits(outputs, target, reduction='none').sum(dim=1)
                        l_seg = l_seg.view(bs, -1).mean(dim=-1)
                        l_seg = self.opts.l_seg * (batch_weight * l_seg).sum() / (batch_weight.sum() + 1e-5)
                        
                        # Self-training Loss for the localizer
                        l_cls = balanced_mask_loss_ce(int_masks_raw, pseudo_gt_seg, l1h, new_classes_mask=new_classes_mask)

                    loss = l_seg + l_cam_out
                    l_reg = l_cls + l_cam_int

                # xxx first backprop of previous loss (compute the gradients for regularization methods)
                # total loss = seg-loss + lkd-loss + encoder-distillation-loss + icarl-loss + cam-loss and pl-seg-loss on localizer
                loss_tot = loss + lkd + lde + l_icarl + l_reg

                # for metrics
                outputs = F.interpolate(outputs, size=images.shape[-2:], mode="bilinear", align_corners=False) # shape B x nb_classes x H x W. upsampled.
                _, prediction = outputs.max(dim=1)  # B, H, W
                prediction = prediction.cpu().numpy()
                labels = labels.cpu().numpy()
                if metrics is not None:
                    metrics.update(labels, prediction)

            self.scaler.scale(loss_tot).backward()
            self.scaler.step(optim)
            if scheduler is not None:
                scheduler.step()
            self.scaler.update()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if tq is not None:
                tq.update(1)
                posftfix_dict = {'Total Loss': f"{loss_tot: .4f}", 'Sem Loss': f"{l_sem_sim:.4f}"}
                tq.set_postfix(posftfix_dict)

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.debug(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                             f" Loss={interval_loss}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss/tot', interval_loss, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_int', l_cam_int, x, intermediate=True)
                    logger.add_scalar('Loss/Loc_loss', l_loc, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_loss', l_cam_new, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_int', l_cls, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_out', l_seg, x, intermediate=True)
                    logger.commit(intermediate=True)
                interval_loss = 0.0

        if tq is not None:
            tq.close()

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        # collect statistics from multiple processes
        if metrics is not None:
            metrics.synch(self.device)

        return (epoch_loss, reg_loss)
    
    def observe_pl_argmax(self, loc_logits, l1h, out_size):
        # return a full sized prediction map

        loc_logits = F.interpolate(loc_logits, size=out_size, mode="bilinear", align_corners=False)
        loc_logits = F.softmax(loc_logits, dim=1)
        loc_logits[:, 1:] *= l1h[:, :, None, None]
        pseudo_gt_seg_lx = binarize(loc_logits)
        pseudo_gt_seg_lx = (self.opts.alpha * pseudo_gt_seg_lx) + \
                                           ((1-self.opts.alpha) * loc_logits)
        return pseudo_gt_seg_lx

    def eval_external(self, loader):
        """
        Evaluate the model predictions on external data set
        """
        device = self.device
        model = self.model
        localizer = self.pseudolabeler
        model_old = self.model_old
        model.eval()
        localizer.eval()
        model_old.eval()

        ret_samples = []

        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                images = x.to(device, dtype=torch.float32)

                with amp.autocast():
                    # prediction from the online model
                    outputs, features = model(images)
                    # prediction from the localizer
                    localizer_logits = localizer(features['body'])
                    # prediction from the old model
                    outputs_old, _ = model_old(images)
                
                _, prediction = F.softmax(outputs, dim=1).max(dim=1)
                _, prediction_old = F.softmax(outputs_old, dim=1).max(dim=1)
                localizer_logits = F.interpolate(localizer_logits, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
                _, localizer_preds = F.softmax(localizer_logits, dim=1).max(dim=1)

                if len(ret_samples) <= 2*self.sample_num:
                    for j, image in enumerate(images):
                        ret_samples.append((
                            images[j].cpu(),
                            prediction[j].cpu(),
                            localizer_preds[j].cpu(),
                            prediction_old[j].cpu()
                        ))
                else:
                    return ret_samples
        return ret_samples
    
    def get_similarity_maps(self, prediction, affinity_matrix=None):
        affinity_matrix = affinity_matrix[self.order, :]
        affinity_matrix = affinity_matrix[:, self.order]

        h, w = prediction.shape
        #similarity_mask = torch.zeros((self.tot_classes - self.old_classes, h, w))
        prediction_1h = F.one_hot(prediction, num_classes=self.old_classes).permute(2, 0, 1).float() # nb_old x h x w
        aff_matrix = affinity_matrix[np.arange(self.old_classes, self.tot_classes), :]
        aff_matrix = aff_matrix[:, np.arange(self.old_classes)] # nb_new x nb_old
        similarity_mask = torch.bmm(torch.from_numpy(aff_matrix).unsqueeze(0).float(), prediction_1h.view(-1, h*w).unsqueeze(0)).reshape(-1, h, w)
        return similarity_mask
    
    def validate_and_visualize(self, loader, metrics, affinity_matrix=None, class_dict=None, save_folder=None, visualize=True):
        # Validate and generate visualizations for every sample
        # only used if the code is run in inference mode
        metrics.reset()
        device = self.device
        
        self.model.eval()
        self.model_old.eval()
        self.pseudolabeler.eval()

        if distributed.get_rank() == 0:
            tq = tqdm.tqdm(total=len(loader))
            tq.set_description("Running inference")
        else:
            tq = None

        with torch.no_grad():
            for i, x in enumerate(loader):
                images = x[0].to(device, dtype=torch.float32)
                labels = x[1].to(device, dtype=torch.long)

                with amp.autocast():
                    # prediction from the online/current model
                    outputs, features = self.model(images)
                    # prediction from the old model
                    outputs_old, _ = self.model_old(images)
                    # prediction from the localizer
                    localizer_logits = self.pseudolabeler(features['body'])

                probabilities, prediction = F.softmax(outputs, dim=1).max(dim=1)
                probabilities_old, prediction_old = F.softmax(outputs_old, dim=1).max(dim=1)
                localizer_masks = F.interpolate(
                    localizer_logits, 
                    size=prediction.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                localizer_probs, localizer_preds = F.softmax(localizer_masks, dim=1).max(dim=1)

                output = {}
                output['image'] = images.cpu()
                output['label'] = labels.cpu()
                output['loc_pred'] = localizer_preds.cpu()
                output['main_pred'] = prediction.cpu()
                output['old_pred'] = prediction_old.cpu()
                output['similarity_map'] = self.get_similarity_maps(prediction_old.squeeze(0).cpu(), affinity_matrix) \
                                            if affinity_matrix is not None else None
            
                if visualize and distributed.get_rank() == 0:
                    visualize_predictions(
                        dataset=self.viz_dataset, 
                        output=output, 
                        mapping_dict=class_dict, 
                        base_path=save_folder,
                        idx=i
                    )
                
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if tq is not None:
                    tq.update(1)
            
            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            if tq is not None:
                tq.close()

        return score, None

    def validate(self, loader, metrics, evaluate_old_model=False, affinity_matrix=None, class_dict=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model_old if evaluate_old_model else self.model
        if self.opts.step > 0:
            model_old = self.model_old
        device = self.device
        ret_samples = []


        model.eval()
        
        if self.opts.step > 0:
            self.pseudolabeler.eval()
            model_old.eval()

        with torch.no_grad():
            for i, x in enumerate(loader):
                images = x[0].to(device, dtype=torch.float32)
                labels = x[1].to(device, dtype=torch.long)
                l1hs = x[2]

                # if self.weakly:
                #     l1h = x[2]

                with amp.autocast():
                    # prediction from the online model
                    outputs, features = model(images)
                    if self.opts.step > 0:
                        # prediction from the old model
                        outputs_old, _ = model_old(images)
                        # prediction from the localizer
                        localizer_logits = self.pseudolabeler(features['body'])
                
                probabilities, prediction = F.softmax(outputs, dim=1).max(dim=1)
                
                if self.opts.step > 0:
                    probabilities_old, prediction_old = F.softmax(outputs_old, dim=1).max(dim=1)
                    fg_mask = 1 - (prediction_old > 0).float()
                    
                    pseudo_gt_seg_lx = self.observe_pl_argmax(localizer_logits, l1hs.to(device, dtype=torch.long), out_size=prediction.shape[-2:])
                    # full res output
                    target_old = torch.sigmoid(outputs_old)
                    target_all = torch.cat((target_old, pseudo_gt_seg_lx[:, self.old_classes:]), dim=1)
                    target_all[:, 0] = torch.min(target_all[:, 0], pseudo_gt_seg_lx[:, 0]) # original is torch.min
                    target_all = target_all.max(dim=1)[1]

                    localizer_masks = F.interpolate(localizer_logits, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
                    localizer_probs, localizer_preds = F.softmax(localizer_masks, dim=1).max(dim=1)

                    new_cls_l1h = nn.functional.one_hot(torch.from_numpy(np.arange(self.old_classes, self.tot_classes)), num_classes=self.tot_classes).sum(dim=0).float()
                    if len(ret_samples) <= self.sample_num:
                        for j, l1h in enumerate(l1hs):
                            # check if the image contains one of the new classes
                            label_present = (new_cls_l1h[1:] * l1h).sum() > 0
                            if label_present:
                                ## similarity mask is used to visualize the similarity masks
                                ## comment out temporarily
                                #similarity_mask = self.get_similarity_maps(prediction_old[j].cpu(), affinity_matrix)
                                ret_samples.append(
                                    (
                                        images[j].cpu(), 
                                        labels[j].cpu(), 
                                        prediction[j].cpu(), 
                                        localizer_preds[j].cpu(), 
                                        prediction_old[j].cpu(),
                                        target_all[j].cpu(),
                                        probabilities_old[j].cpu(),
                                        localizer_probs[j].cpu(),
                                        fg_mask[j].cpu(),
                                        #similarity_mask.cpu()
                                    )
                                )

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score, ret_samples

    def validate_CAM(self, loader, metrics):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device

        self.pseudolabeler.eval()
        model.eval()

        def classify(images):
            masks = self.pseudolabeler(model(images, as_feature_extractor=True)['body'])
            masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
            masks = masks.softmax(dim=1)
            return masks

        i = -1
        with torch.no_grad():
            for x in tqdm.tqdm(loader):
                i = i+1
                images = x[0].to(device, dtype=torch.float32)
                labels = x[1].to(device, dtype=torch.long)
                l1h = x[2].to(device, dtype=torch.bool)

                with amp.autocast():
                    masks = classify(images)

                _, prediction = masks.max(dim=1)

                labels[labels < self.old_classes] = 0
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score
    
    def load_ckpt_inference(self, prev_path, curr_path):
        # a helper function to load the current and previous checkpoints.
        # mainly needed for custom visualizations
        if osp.exists(prev_path) and osp.exists(curr_path):
            curr_step_checkpoint = torch.load(curr_path, map_location='cpu')
            self.model.load_state_dict(curr_step_checkpoint['model_state'], strict=True)
            self.pseudolabeler.load_state_dict(curr_step_checkpoint['pseudolabeler'], strict=True)

            # Load state dict from the model state dict, that contains the old model parameters
            step_checkpoint = torch.load(prev_path, map_location="cpu")
            new_state = {}
            for k, v in step_checkpoint['model_state'].items():
                new_state[k[7:]] = v
            self.model_old.load_state_dict(new_state, strict=True)  # Load also here old parameters

            self.logger.info(f"[!] Current model loaded from {curr_path}")
            self.logger.info(f"[!] Previous model loaded from {prev_path}")

            # clean memory
            del step_checkpoint['model_state']
            del curr_step_checkpoint['model_state']
        else:
            raise FileNotFoundError(f'Either {prev_path} or {curr_path} does not exist!')

    def load_step_ckpt(self, path):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            if self.opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                self.model.module.init_new_classifier(self.device)
            # Load state dict from the model state dict, that contains the old model parameters
            new_state = {}
            for k, v in step_checkpoint['model_state'].items():
                new_state[k[7:]] = v
            self.model_old.load_state_dict(new_state, strict=True)  # Load also here old parameters

            self.logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif self.opts.debug:
            self.logger.info(f"[!] WARNING: Unable to find of step {self.opts.step - 1}! "
                             f"Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)

    def load_ckpt(self, path):
        opts = self.opts
        assert osp.isfile(path), f"Error, ckpt not found in {path}"

        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.weakly:
            self.pseudolabeler.load_state_dict(checkpoint["pseudolabeler"])

        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        self.logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        del checkpoint

        return cur_epoch, best_score
