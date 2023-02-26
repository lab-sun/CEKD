import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    # https://github.com/gasparian/PicsArtHack-binary-segmentation/blob/ecab001f334949d5082a79b8fbd1dc2fdb8b093e/utils.py#L217

    def __init__(self, loss_weight = 1) -> None:
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight

    def auto_weight_bce(self, y_hat_log, y):
        if y.ndim == 3:
            y = y.unsqueeze(1)
        with torch.no_grad():
            beta = y.mean(dim=[2, 3], keepdims=True)
        logit_1 = F.logsigmoid(y_hat_log)
        logit_0 = F.logsigmoid(-y_hat_log)
        loss = -(1 - beta) * logit_1 * y - beta * logit_0 * (1 - y)
        return loss.mean()

    def forward(self, pred_class_map,target_maps,pred_edge_map,target_edges) -> torch.Tensor:
        
        loss1 = self.loss_weight * (self.auto_weight_bce(pred_edge_map, target_edges.float()))
        loss2 = F.cross_entropy(pred_class_map, target_maps)
        print("loss_weight: {} loss1(edge): {}  loss2(segmentation): {} ".format(self.loss_weight, loss1, loss2))
        return loss1 + loss2 ,loss1,loss2
