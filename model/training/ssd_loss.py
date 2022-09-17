# inspired by: https://github.com/lufficc/SSD
import torch
from torch import nn
import torch.nn.functional as F
import math

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    Args:
        loss (N, n): the loss for each example.
        labels (N, n): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    # Even if there's no positive example for the class, generate 1 * neg_pos_ratio negatives
    num_pos[num_pos == 0] = 1
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask, torch.sum(num_pos).item()


class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        super(SSDLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, predictions, targets):
        """Compute classification loss and smooth l1 loss.
         input: confidence, predicted_locations, labels, gt_locations
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """

        player_loc, player_conf = predictions
        player_loc_t, player_conf_t = targets
        batch_size = player_loc.shape[0]

        player_loc = player_loc.view(batch_size, -1, 4)
        player_conf = player_conf.view(batch_size, -1, 2)
        player_loc_t = player_loc_t.view(batch_size, -1, 4)
        player_conf_t = player_conf_t.view(batch_size, -1)

        with torch.no_grad():
            loss_player = -F.log_softmax(player_conf, dim=2)[:, :, 0]
            mask_player, num_pos_player = hard_negative_mining(loss_player, player_conf_t, self.neg_pos_ratio)

        player_conf = player_conf[mask_player, :]

        player_loss_c = F.cross_entropy(player_conf, player_conf_t[mask_player], reduction='sum')

        pos_mask_player = player_conf_t > 0
        player_loc = player_loc[pos_mask_player, :].view(-1, 4)
        player_loc_t = player_loc_t[pos_mask_player, :].view(-1, 4)
        player_loss_l = F.smooth_l1_loss(player_loc, player_loc_t, reduction='sum')

        return player_loss_l / num_pos_player, player_loss_c / num_pos_player
