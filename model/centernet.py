import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from wtw.dataset import WTWDataset
from loss.losses import modified_focal_loss
from .backbone.renset import ResNet
from model.decoder import Decoder
from model.head import Head
from model.fpn import FPN


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


class CenterNet(pl.LightningModule):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.backbone = ResNet(cfg.slug)
        if cfg.fpn:
            self.fpn = FPN(self.backbone.outplanes)
        self.upsample = Decoder(self.backbone.outplanes if not cfg.fpn else 2048, cfg.bn_momentum)
        self.head = Head(channel=cfg.head_channel, num_classes=cfg.num_classes)

        self._fpn = cfg.fpn
        self.down_stride = cfg.down_stride
        self.score_th = cfg.score_th
        self.CLASSES_NAME = cfg.CLASSES_NAME

        self.dataset_root = cfg.root
        self.resize_size = cfg.resize_size
        self.batch_size = cfg.batch_size

        self.focal_loss = modified_focal_loss
        self.l1_loss = nn.L1Loss()

    def forward(self, x):
        feats = self.backbone(x)
        if self._fpn:
            feat = self.fpn(feats)
        else:
            feat = feats[-1]
        return self.head(self.upsample(feat))

    @torch.no_grad()
    def inference(self, img, infos, topK=40, return_hm=False, th=None):
        feats = self.backbone(img)
        if self._fpn:
            feat = self.fpn(feats)
        else:
            feat = feats[-1]
        pred_hm, pred_dm, pred_v2c, pred_c2v = self.head(self.upsample(feat))

        _, _, h, w = img.shape
        b, c, output_h, output_w = pred_hm.shape
        pred_hm = self.pool_nms(pred_hm)
        scores, index, clses, ys, xs = self.topk_score(pred_hm, K=topK)
        #
        # reg = gather_feature(pred_offset, index, use_transform=True)
        # reg = reg.reshape(b, topK, 2)
        # xs = xs.view(b, topK, 1) + reg[:, :, 0:1]
        # ys = ys.view(b, topK, 1) + reg[:, :, 1:2]
        #
        # wh = gather_feature(pred_wh, index, use_transform=True)
        # wh = wh.reshape(b, topK, 2)

        clses = clses.reshape(b, topK, 1).float()
        scores = scores.reshape(b, topK, 1)

        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
        bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)

        detects = []
        for batch in range(b):
            mask = scores[batch].gt(self.score_th if th is None else th)

            batch_boxes = bboxes[batch][mask.squeeze(-1), :]
            # batch_boxes[:, [0, 2]] *= infos[batch]['raw_width'] / output_w
            # batch_boxes[:, [1, 3]] *= infos[batch]['raw_height'] / output_h
            batch_boxes[:, [0, 2]] *= w / output_w
            batch_boxes[:, [1, 3]] *= h / output_h

            batch_scores = scores[batch][mask]

            batch_clses = clses[batch][mask]
            batch_clses = [self.CLASSES_NAME[int(cls.item())] for cls in batch_clses]

            detects.append([batch_boxes, batch_scores, batch_clses, pred_hm[batch] if return_hm else None])
        return detects

    def training_step(self, batch, batch_idx):
        image, (gt_hm, gt_dm, gt_v2c, gt_c2v) = batch
        pred_hm, pred_dm, pred_v2c, pred_c2v = self.forward(image)

        hm_loss = self.focal_loss(pred_hm, gt_hm)
        dm_loss = self.l1_loss(pred_dm, gt_dm)
        v2c_loss = self.l1_loss(pred_v2c, gt_v2c)
        c2v_loss = self.l1_loss(pred_c2v, gt_c2v)

        loss = hm_loss + dm_loss + v2c_loss + c2v_loss
        self.logger.experiment.add_scalar("hm_loss/train", hm_loss, self.global_step)
        self.logger.experiment.add_scalar("dm_loss/train", dm_loss, self.global_step)
        self.logger.experiment.add_scalar("v2c_loss/train", v2c_loss, self.global_step)
        self.logger.experiment.add_scalar("c2v_loss/train", c2v_loss, self.global_step)
        self.logger.experiment.add_scalar("total_loss/train", loss, self.global_step)
        return loss

    def pool_nms(self, hm, pool_size=3):
        pad = (pool_size - 1) // 2
        hm_max = F.max_pool2d(hm, pool_size, stride=1, padding=pad)
        keep = (hm_max == hm).float()
        return hm * keep

    def topk_score(self, scores, K):
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        plateau_lr = ReduceLROnPlateau(optimizer, patience=5000, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": plateau_lr,
                "monitor": "train_loss",
                'interval': "step",
                "frequency": 1,
            }
        }

    def train_dataloader(self):
        wtw_dataset = WTWDataset(
                        self.dataset_root+'xml/',
                        self.dataset_root + 'images/',
                        self.resize_size,
        )
        return DataLoader(wtw_dataset, batch_size=self.batch_size, shuffle=True, num_workers=64)

    def val_dataloader(self):
        wtw_dataset = WTWDataset(
                        self.dataset_root+'xml/',
                        self.dataset_root + 'images/',
                        self.resize_size,
        )
        return DataLoader(wtw_dataset, batch_size=self.batch_size, num_workers=64)
