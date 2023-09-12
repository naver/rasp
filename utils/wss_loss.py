import torch.nn.functional as F
import torch
from torch import distributed
import numpy as np


def ngwp_focal(outputs, outputs_old, n_cls, focal=True, alpha=1e-5, lam=1e-2):
    bs, c, h, w = outputs.size()
    
    masks = F.softmax(outputs, dim=1)
    
    # code remains the same
    masks_ = masks.view(bs, c, -1)
    logits = outputs.view(bs, c, -1)
    # y_ngwp = (logits * (masks_ + alpha)).sum(-1) / (masks_+alpha).sum(-1)
    y_ngwp = (logits * masks_).sum(-1) / (1.0 + masks_.sum(-1))

    # focal penalty loss
    if focal:
        y_focal = torch.pow(1 - masks_.mean(-1), 3) * torch.log(lam + masks_.mean(-1))
        y = y_ngwp + y_focal
    else:
        y = y_ngwp
    return y


def attention_cam(outputs, outputs_old, n_cls, alpha=0.01):
    bs, c, h, w = outputs.size()

    masks = F.softmax(outputs, dim=1)
    masks_ = masks.view(bs, c, -1)
    logits = outputs.view(bs, c, -1)

    y_ngwp = (logits * (masks_ + alpha)).sum(-1) / (masks_+alpha).sum(-1)
    return y_ngwp

def sem_bce_loss(outputs, labels, outputs_old, nb_new_classes, affinity_matrix, order):
    """
    It implements the semantic prior loss.
    Computes a BCE loss between the localizer predictions (for new class logits) and 
    the semantic similarity maps computed through the old model
    """

    affinity_matrix = affinity_matrix[order, :]
    affinity_matrix = affinity_matrix[:, order]

    bs, c, h, w = outputs.size()
    nb_old_classes = outputs_old.shape[1]

    # create the semantic maps
    preds_old = F.softmax(outputs_old.detach(), dim=1).max(dim=1)[1]
    preds_old_1h = F.one_hot(preds_old, num_classes=nb_old_classes).permute(0, 3, 1, 2).float() # bs x nb_old x h x w
    aff_matrix = affinity_matrix[np.arange(nb_old_classes, nb_old_classes + nb_new_classes), :]
    aff_matrix = torch.from_numpy(aff_matrix[:, np.arange(nb_old_classes)]).unsqueeze(0).repeat(bs, 1, 1).float().cuda() # bs x nb_new x nb_old
    similarity_weights = torch.bmm(aff_matrix, preds_old_1h.view(bs, nb_old_classes, h*w)).reshape(bs, nb_new_classes, h, w) # bs x nb_new x h x w
    # normalize the weights so that the background pixels in each channel has a weight = 1
    aff_bkg = torch.from_numpy(affinity_matrix[0, nb_old_classes:nb_old_classes + nb_new_classes])
    weight_bkg = (aff_bkg * torch.ones((nb_new_classes), dtype=torch.float)).view(1, nb_new_classes, 1, 1).cuda() # 1 x nb_new x 1 x 1
    similarity_weights = similarity_weights / weight_bkg
    norm_similarity_weights = F.sigmoid(similarity_weights)

    l = F.binary_cross_entropy_with_logits(outputs[:, -nb_new_classes:], norm_similarity_weights, reduction='none') # bs x nb_new x h x w
    l = (l * labels[:, -nb_new_classes:].view(bs, nb_new_classes, 1, 1)).mean()

    return l, norm_similarity_weights

def sem_ngwp_focal(outputs, outputs_old, nb_new_classes, focal=True, alpha=1e-5, lam=1e-2, 
                   affinity_matrix=None):
    """
    Same as ngwp_focal loss except, it also considers the semantic masks obtained with the semantic similarity
    metric
    """
    bs, c, h, w = outputs.size()
    nb_old_classes = outputs_old.shape[1]
    masks = F.softmax(outputs, dim=1)

    # create the semantic maps
    preds_old = F.softmax(outputs_old.detach(), dim=1).max(dim=1)[1]
    preds_old_1h = F.one_hot(preds_old, num_classes=nb_old_classes).permute(0, 3, 1, 2).float() # bs x nb_old x h x w
    aff_matrix = affinity_matrix[np.arange(nb_old_classes, nb_old_classes + nb_new_classes), :]
    aff_matrix = torch.from_numpy(aff_matrix[:, np.arange(nb_old_classes)]).unsqueeze(0).repeat(bs, 1, 1).float().cuda() # bs x nb_new x nb_old
    similarity_weights = torch.bmm(aff_matrix, preds_old_1h.view(bs, nb_old_classes, h*w)).reshape(bs, nb_new_classes, h, w) # bs x nb_new x h x w
    # normalize the weights so that the background pixels in each channel has a weight = 1
    aff_bkg = torch.from_numpy(affinity_matrix[0, nb_old_classes:nb_old_classes + nb_new_classes])
    weight_bkg = (aff_bkg * torch.ones((nb_new_classes), dtype=torch.float)).view(1, nb_new_classes, 1, 1).cuda() # 1 x nb_new x 1 x 1
    norm_similarity_weights = similarity_weights / weight_bkg

    # re-weight the masks with the similarity weights
    weights = torch.cat([torch.ones((bs, nb_old_classes, h, w), dtype=torch.float).cuda(), norm_similarity_weights], dim=1)
    res_masks = masks * weights

    masks = (masks + res_masks) / 2

    masks_ = masks.view(bs, c, -1)
    logits = outputs.view(bs, c, -1)
    # y_ngwp = (logits * (masks_ + alpha)).sum(-1) / (masks_+alpha).sum(-1)
    y_ngwp = (logits * masks_).sum(-1) / (1.0 + masks_.sum(-1))

    # focal penalty loss
    if focal:
        y_focal = torch.pow(1 - masks_.mean(-1), 3) * torch.log(lam + masks_.mean(-1))
        y = y_ngwp + y_focal
    else:
        y = y_ngwp
    return y


def bce_loss(outputs, outputs_old, nb_new_classes, labels, mode='ngwp', reduction='sum', mask=None, affinity_matrix=None):
    
    # this function computes the loss to train the auxiliary classifier on image labels
    # it uses BCE loss to train the classifier.
    
    # input params
    # outputs: B x tot_classes x h x w, where h and w are the downsampled output height and width.
    # labels:  B x new_classes OR tot_classes depending upon the --no_mask flag
    
    n_cls = labels.shape[-1]
    bs, c, h, w = outputs.size()

    if mode == 'ngwp':
        y = ngwp_focal(outputs, outputs_old, nb_new_classes)
    elif mode == 'att':
        y = attention_cam(outputs, outputs_old, nb_new_classes)
    elif mode == 'sem-ngwp':
        y = sem_ngwp_focal(outputs, outputs_old, nb_new_classes, affinity_matrix=affinity_matrix)
    else:
        logits = outputs.view(bs, c, -1) # bs x c x h*w
        y = logits.mean(-1) # bs x c

    #bs, n_cls = labels.shape
    
    if reduction == 'sum':
        y = y[:, -n_cls:] # last n_cls elements in the logits that correspond to the new classes
        l = F.binary_cross_entropy_with_logits(y, labels, reduction="none").sum(dim=1).mean()
    elif reduction == 'masked-sum':
        # this is for the case when the bkg is also considered for the WS loss
        labels = torch.cat([torch.ones(bs, 1).cuda(), labels], dim=1)
        mask = mask.unsqueeze(0).repeat(bs, 1).cuda()
        mask[:, 0] = 1.
        l = (F.binary_cross_entropy_with_logits(y, labels, reduction="none") * mask).sum(dim=1).mean()
    else:
        y = y[:, -n_cls:] # last n_cls elements in the logits that correspond to the new classes
        l = F.binary_cross_entropy_with_logits(y, labels)
    return l


def ce_loss(inputs, labels):
    outputs = torch.zeros_like(inputs[:, 0])  # B, 1, H, W
    den = torch.logsumexp(inputs, dim=1)      # B, H, W       den of softmax

    for i in range(len(outputs)):
        new_lbl = labels[i].nonzero()[:, 0] + 1
        new_lbl = torch.cat([torch.tensor([0], device=new_lbl.device), new_lbl])
        to_sum = inputs[i, new_lbl]
        outputs[i] = torch.logsumexp(to_sum, dim=0) - den[i]  # H, W       p(O)

    loss = - outputs.mean()

    return loss


def ce_penalty_loss(inputs, labels):
    bs, c, h, w = inputs.size()
    outputs = torch.zeros_like(inputs[:, 0])  # B, 1, H, W
    penalty = torch.zeros(bs, device=inputs.device)

    den = torch.logsumexp(inputs, dim=1)      # B, H, W       den of softmax

    masks = F.softmax(inputs, dim=1)
    masks_ = masks.view(bs, c, -1)

    for i in range(len(outputs)):
        new_lbl = labels[i].nonzero()[:, 0] + 1
        penalty[i] = (torch.pow(1 - masks_[i].mean(-1), 3) * torch.log(0.01 + masks_[i].mean(-1)))[new_lbl].mean()
        new_lbl = torch.cat([torch.tensor([0], device=new_lbl.device), new_lbl])
        to_sum = inputs[i, new_lbl]
        outputs[i] = torch.logsumexp(to_sum, dim=0) - den[i]  # H, W       p(O)

    loss = - (penalty.mean() + outputs.mean())
    return loss


def eps_loss(cam, cam2, label, tau=0.4, lam=0.5, intermediate=False):
    """
    Get EPS loss for pseudo-pixel supervision from saliency map.
    Args:
        cam (tensor): response from model with float values.
        saliency (tensor): saliency map from off-the-shelf saliency model.
        num_classes (int): the number of classes
        label (tensor): label information.
        tau (float): threshold for confidence area
        lam (float): blending ratio between foreground map and background map
        intermediate (bool): if True return all the intermediates, if not return only loss.
    Shape:
        cam (N, C, H', W') where N is the batch size and C is the number of classes.
        saliency (N, 1, H, W)
        label (N, C)
    """
    b, c, h, w = cam.size()
    num_classes = c - 1

    cam2 = cam2.softmax(dim=1)  # B x Co x H x W
    # cam2 = F.interpolate(cam2, size=(h, w))  # interpolate cam2 to higher dimension -> here they are sigmoid logits
    cam2 = (cam2.detach() > 0.5).type_as(cam2)  # round them based on 0.5 -> higher is 1, lower is 0
    cam2_fg = (cam2[:, :1].sum(dim=1) > 0).type_as(cam2)  # compute if class is present (at least 1) then FG
    saliency = cam2_fg * lam + (1 - cam2[:, 0]) * (1 - lam)  # B x H x W
    saliency.unsqueeze_(dim=1)  # B x 1 x H x W

    label_map = label.view(b, num_classes, 1, 1).expand(size=(b, num_classes, h, w)).bool()

    # Map selection
    label_map_fg = torch.zeros(size=(b, num_classes + 1, h, w)).bool().cuda()
    label_map_bg = torch.zeros(size=(b, num_classes + 1, h, w)).bool().cuda()

    label_map_bg[:, 0] = True  # bkg is in any image
    label_map_fg[:, 1:] = label_map.clone()  # 1s only for classes in images

    sal_pred = F.softmax(cam, dim=1)  # make CAMs -> N, C, H, W

    iou_saliency = (torch.round(sal_pred[:, 1:].detach()) * torch.round(saliency)).view(b, num_classes, -1).sum(-1) / \
                   (torch.round(sal_pred[:, 1:].detach()) + 1e-04).view(b, num_classes, -1).sum(-1)

    valid_channel = (iou_saliency > tau).view(b, num_classes, 1, 1).expand(size=(b, num_classes, h, w))

    label_map_fg[:, 1:] = label_map & valid_channel
    label_map_bg[:, 1:] = label_map & (~valid_channel)

    # Saliency loss
    fg_map = torch.zeros_like(sal_pred).cuda()
    bg_map = torch.zeros_like(sal_pred).cuda()

    fg_map[label_map_fg] = sal_pred[label_map_fg]
    bg_map[label_map_bg] = sal_pred[label_map_bg]

    fg_map = torch.sum(fg_map, dim=1, keepdim=True)
    bg_map = torch.sum(bg_map, dim=1, keepdim=True)

    bg_map = torch.sub(1, bg_map)
    sal_pred = fg_map * lam + bg_map * (1 - lam)

    loss = F.mse_loss(sal_pred, saliency)

    if intermediate:
        return loss, fg_map, bg_map, sal_pred
    else:
        return loss


def eps_loss_cw(cam, cam2, label, tau=0.4):
    b, c, h, w = cam.size()
    c2 = cam2.shape[1]

    label = label.bool()  # .expand(size=(b, num_classes, h, w)).bool()
    saliency_pred = torch.zeros(size=(b, h, w)).to(cam.device)
    saliency = torch.zeros(size=(b, h, w)).bool().to(cam.device)

    cam2 = cam2.softmax(dim=1)  # B x Co x H x W
    saliency_pc = (cam2[:, 1:].detach() > 0.5).bool()
    # round them based on 0.5 -> higher is 1, lower is 0

    sal_pred = F.softmax(cam, dim=1)  # make CAMs -> N, C, H, W
    sal_pred_bin = (sal_pred[:, 1:].detach() > 0.5)

    for j in range(0, c-1):
        s_c = sal_pred_bin[:, j].unsqueeze(1).bool().detach()
        # returns a value of IoU between class C and prior classes
        iou_saliency = (s_c * saliency_pc).view(b, c2-1, -1).sum(-1) / \
                   (s_c.bitwise_or(saliency_pc)).view(b, c2-1, -1).sum(-1)
        # This is 1 for each channel
        valid_channel = (iou_saliency > tau).view(b, c2-1)
        batch_sel = (label[:, j] & (valid_channel.sum(dim=1) > 0))
        saliency_pred += batch_sel.type_as(sal_pred).view(b, 1, 1) * sal_pred[:, j]
        saliency += (valid_channel.view(b, c2-1, 1, 1) * saliency_pc).max(dim=1)[0]

    saliency = saliency.float()
    return F.mse_loss(saliency_pred, saliency)


def binarize(input):
    max = input.max(dim=1, keepdim=True)[0]
    return (input >= max).type_as(input)


def refine_mask(cam_orig, out_old, label, tau=0.5, bin=True):
    b, c_tot, h, w = cam_orig.size()
    c_old = out_old.shape[1]

    label = label.bool()
    out_old = binarize(out_old.detach())[:, 1:].bool()
    cam = binarize(cam_orig.detach())
    if bin:
        cam_orig = cam

    for i in range(b):  # for each image
        for j in range(1, c_tot):
            if label[i, j-1]:
                s_c = cam[i, j].unsqueeze(0).bool()  # 1HW
                # returns the overlap of new CAM on old class
                overlap = (s_c.bitwise_and(out_old[i])).view(c_old - 1, -1).sum(-1) / \
                          ((s_c.bitwise_or(out_old[i])).view(c_old - 1, -1).sum(-1) + 1)
                overlap_classes = (overlap > tau).view(c_old - 1)
                if overlap_classes.sum() > 0:
                    nc = ((out_old[i] * overlap_classes.view(c_old - 1, 1, 1)).sum(dim=0) > 0).type_as(cam)
                    cam_orig[i, j] = nc

    return cam_orig
