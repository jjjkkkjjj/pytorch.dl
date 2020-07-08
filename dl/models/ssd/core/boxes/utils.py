import torch

from .....data.utils.boxes import centroids2corners, iou

def matching_strategy(targets, dboxes, **kwargs):
    """
    :param targets: Tensor, shape is (batch*object num(batch), 1+4+class_labels)
    :param dboxes: shape is (default boxes num, 4)
    IMPORTANT: Note that means (cx, cy, w, h)
    :param kwargs:
        threshold: (Optional) float, threshold for returned indicator
        batch_num: (Required) int, batch size
    :return:
        pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        matched_targets: Tensor, shape = (batch, default box num, 4+class_num) including background
    """
    threshold = kwargs.pop('threshold', 0.5)
    batch_num = kwargs.pop('batch_num')
    device = dboxes.device



    dboxes_num = dboxes.shape[0]
    # minus 'box number per image' and 'localization=(cx, cy, w, h)'
    class_num = targets[0].shape[1] - 4

    # convert centered coordinated to minmax coordinates
    dboxes_mm = centroids2corners(dboxes)

    # create returned empty Tensor
    pos_indicator, matched_targets = torch.empty((batch_num, dboxes_num), device=device, dtype=torch.bool), torch.empty((batch_num, dboxes_num, 4 + class_num), device=device)

    # matching for each batch
    index = 0
    for b, target in enumerate(targets):
        targets_loc, targets_conf = target[:, :4], target[:, 4:]

        # overlaps' shape = (object num, default box num)
        overlaps = iou(centroids2corners(targets_loc), dboxes_mm.clone())
        """
        best_overlap_per_object, best_dbox_ind_per_object = overlaps.max(dim=1)
        best_overlap_per_dbox, best_object_ind_per_dbox = overlaps.max(dim=0)
        for object_ind, dbox_ind in enumerate(best_dbox_ind_per_object):
            best_object_ind_per_dbox[dbox_ind] = object_ind
        best_overlap_per_dbox.index_fill_(0, best_dbox_ind_per_object, 999)

        pos_ind = best_overlap_per_dbox > threshold
        pos_indicator[b] = pos_ind
        gt_loc[b], gt_conf[b] = targets[best_object_ind_per_dbox], targets_conf[best_object_ind_per_dbox]

        neg_ind = torch.logical_not(pos_ind)
        gt_conf[b, neg_ind] = 0
        gt_conf[b, neg_ind, -1] = 1
        """
        # get maximum overlap value for each default box
        # shape = (batch num, dboxes num)
        overlaps_per_dbox, object_indices = overlaps.max(dim=0)
        #object_indices = object_indices.long() # for fancy indexing

        # get maximum overlap values for each object
        # shape = (batch num, object num)
        overlaps_per_object, dbox_indices = overlaps.max(dim=1)
        for obj_ind, dbox_ind in enumerate(dbox_indices):
            object_indices[dbox_ind] = obj_ind
        overlaps_per_dbox.index_fill_(0, dbox_indices, threshold + 1)# ensure N!=0

        pos_ind = overlaps_per_dbox > threshold

        # assign targets
        matched_targets[b, :, :4], matched_targets[b, :, 4:] = targets_loc[object_indices], targets_conf[object_indices]
        pos_indicator[b] = pos_ind

        # set background flag
        neg_ind = torch.logical_not(pos_ind)
        matched_targets[b, neg_ind, 4:] = 0
        matched_targets[b, neg_ind, -1] = 1



    return pos_indicator, matched_targets



def matching_strategy_quads(targets, dboxes, **kwargs):
    """
    :param targets: Tensor, shape is (batch*object num(batch), 4=(cx,cy,w,h)+8=(x1,y1,x2,y2,...)+class_labels)
    :param dboxes: shape is (default boxes num, 4)
    IMPORTANT: Note that means (cx, cy, w, h)
    :param kwargs:
        threshold: (Optional) float, threshold for returned indicator
        batch_num: (Required) int, batch size
    :return:
        pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        matched_targets: Tensor, shape = (batch, default box num, 4+class_num) including background
    """
    threshold = kwargs.pop('threshold', 0.5)
    batch_num = kwargs.pop('batch_num')
    device = dboxes.device

    dboxes_num = dboxes.shape[0]
    # minus 'box number per image' and 'localization=(cx, cy, w, h)'
    class_num = targets[0].shape[1] - 4 - 8

    # convert centered coordinated to minmax coordinates
    dboxes_mm = centroids2corners(dboxes)

    # create returned empty Tensor
    pos_indicator, matched_targets = torch.empty((batch_num, dboxes_num), device=device, dtype=torch.bool), torch.empty(
        (batch_num, dboxes_num, 4 + 8 + class_num), device=device)

    # matching for each batch
    index = 0
    for b, target in enumerate(targets):
        targets_loc, targets_quad, targets_conf = target[:, :4], target[:, 4:12], target[:, 12:]

        # overlaps' shape = (object num, default box num)
        overlaps = iou(centroids2corners(targets_loc), dboxes_mm.clone())
        """
        best_overlap_per_object, best_dbox_ind_per_object = overlaps.max(dim=1)
        best_overlap_per_dbox, best_object_ind_per_dbox = overlaps.max(dim=0)
        for object_ind, dbox_ind in enumerate(best_dbox_ind_per_object):
            best_object_ind_per_dbox[dbox_ind] = object_ind
        best_overlap_per_dbox.index_fill_(0, best_dbox_ind_per_object, 999)

        pos_ind = best_overlap_per_dbox > threshold
        pos_indicator[b] = pos_ind
        gt_loc[b], gt_conf[b] = targets[best_object_ind_per_dbox], targets_conf[best_object_ind_per_dbox]

        neg_ind = torch.logical_not(pos_ind)
        gt_conf[b, neg_ind] = 0
        gt_conf[b, neg_ind, -1] = 1
        """
        # get maximum overlap value for each default box
        # shape = (batch num, dboxes num)
        overlaps_per_dbox, object_indices = overlaps.max(dim=0)
        # object_indices = object_indices.long() # for fancy indexing

        # get maximum overlap values for each object
        # shape = (batch num, object num)
        overlaps_per_object, dbox_indices = overlaps.max(dim=1)
        for obj_ind, dbox_ind in enumerate(dbox_indices):
            object_indices[dbox_ind] = obj_ind
        overlaps_per_dbox.index_fill_(0, dbox_indices, threshold + 1)  # ensure N!=0

        pos_ind = overlaps_per_dbox > threshold

        # assign targets
        matched_targets[b, :, :4], matched_targets[b, :, 4:12], matched_targets[b, :, 12:] = \
            targets_loc[object_indices], targets_quad[object_indices], targets_conf[object_indices]
        pos_indicator[b] = pos_ind

        # set background flag
        neg_ind = torch.logical_not(pos_ind)
        matched_targets[b, neg_ind, 12:] = 0
        matched_targets[b, neg_ind, -1] = 1

    return pos_indicator, matched_targets


