# coding:utf-8
# By Zhen Feng, Feb. 2, 2023
# Email: zfeng94@outlook.com

import numpy as np 
from PIL import Image 
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('runs/Pred_' + weight_name + '_' + image_name[i] + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    F1_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN
        if (recall_per_class[cid] == np.nan) | (precision_per_class[cid] == np.nan) |(precision_per_class[cid]==0)|(recall_per_class[cid]==0):
            F1_per_class[cid] = np.nan
        else :
            F1_per_class[cid] = 2 / (1/precision_per_class[cid] +1/recall_per_class[cid])

    return precision_per_class, recall_per_class, iou_per_class,F1_per_class


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


def kd_ce_loss(logits_S, logits_T, temperature=1):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=1)).sum(dim=1).mean()
    return loss
