from torch import nn

from .ssd import LocalizationLoss, ConfidenceLoss

class TextBoxLoss(nn.Module):
    def __init__(self, alpha=1, loc_loss=None, conf_loss=None):
        super().__init__()

        self.alpha = alpha
        self.loc_loss = LocalizationLoss() if loc_loss is None else loc_loss
        self.conf_loss = ConfidenceLoss() if conf_loss is None else conf_loss

    def forward(self, pos_indicator, predicts, targets):
        """
        :param pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        :param predicts: Tensor, shape is (batch, total_dbox_nums, 4+class_labels=(cx, cy, w, h, p_class,...)
        :param targets: Tensor, shape is (batch, total_dbox_nums, 4+class_labels=(cx, cy, w, h, p_class,...)
        :return:
            loss: float
        """
        # get localization and confidence from predicts and targets respectively
        pred_loc, pred_conf = predicts[:, :, :12], predicts[:, :, 12:]
        targets_loc, targets_conf = targets[:, :, :12], targets[:, :, 12:]

        # Localization loss
        loc_loss = self.loc_loss(pos_indicator, pred_loc, targets_loc)

        # Confidence loss
        conf_loss = self.conf_loss(pos_indicator, pred_conf, targets_conf)

        return conf_loss, loc_loss