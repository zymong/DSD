import torch
import torch.nn as nn
from .base import FlowModel
from .camliga_core import CamLiGA_Core
from .losses import calc_sequence_loss_2d, calc_sequence_loss_3d
from .utils import InputPadder, size_of_batch, knn_interpolation
from .ids import paral2persp, persp2paral
import numpy as np
import cv2


# python eval_things.py testset=flyingthings3d_subset model=camliraft ckpt.path=checkpoints/camliraft_things150e.pt

class CamLiGA(FlowModel):
    def __init__(self, cfgs):
        super(CamLiGA, self).__init__()
        self.cfgs = cfgs
        self.core = CamLiGA_Core(cfgs)

    def train(self, mode=True):
        self.training = mode

        for module in self.children():
            module.train(mode)

        if self.cfgs.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

        return self

    def eval(self):
        return self.train(False)

    def forward(self, inputs):
        images = inputs['images'].float()

        # 具体地说，[:, :3] 选取数组中所有行和前三列的元素，相当于提取出数组中的左半部分。
        # [:, 3:] 则选取所有行和第四至第六列的元素，相当于提取出数组中的右半部分。
        # 代码通过数组切片操作将它分成两个形状分别为 (n, 3) 的子数组 pc1 和 pc2。
        pc1, pc2 = inputs['pcs'][:, :3], inputs['pcs'][:, 3:]
        intrinsics = inputs['intrinsics']

        # pad input shape to a multiple of 8 将输入形状改为8的倍数
        padder = InputPadder(images.shape, x=8)
        image1, image2 = padder.pad(images[:, :3], images[:, 3:])
        # src_image1, src_image2 =  image1, image2

        # 图像像素值标准化,平均值和标准差是预先计算得到的常量值
        # 图像在每个通道上的平均值
        norm_mean = torch.tensor([123.675, 116.280, 103.530], device=images.device)
        # 图像在每个通道上的标准差
        norm_std = torch.tensor([58.395, 57.120, 57.375], device=images.device)
        # reshape将它们扩展为与图像张量具有相同的形状
        # 最后，将 image1 和 image2 分别减去 norm_mean，再除以 norm_std，即可对它们进行标准化处理。
        image1 = image1 - norm_mean.reshape(1, 3, 1, 1)
        image2 = image2 - norm_mean.reshape(1, 3, 1, 1)
        image1 = image1 / norm_std.reshape(1, 3, 1, 1)
        image2 = image2 / norm_std.reshape(1, 3, 1, 1)

        # print(image1.shape)           torch.Size([4, 3, 544, 960])    （4, 3, 544, 960）分别表示（batch size, channel, height, width）

        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': image1.shape[-2],  # 544
            'sensor_w': image1.shape[-1],  # 960
            'f': intrinsics[:, 0],
            'cx': intrinsics[:, 1],
            'cy': intrinsics[:, 2],
        }
        paral_cam_info = {
            'projection_mode': 'parallel',
            'sensor_h': round(image1.shape[-2] / 32),
            'sensor_w': round(image1.shape[-1] / 32),
            'cx': (round(image1.shape[-1] / 32) - 1) / 2,
            'cy': (round(image1.shape[-2] / 32) - 1) / 2,
        }
        
        # 逆深度变换ids
        pc1 = persp2paral(pc1, persp_cam_info, paral_cam_info)
        pc2 = persp2paral(pc2, persp_cam_info, paral_cam_info)


        # ！！计算预测的光流和场景流！！
        flow_2d_preds, flow_3d_preds = self.core(image1, image2, pc1, pc2, paral_cam_info)

        for i in range(len(flow_2d_preds)):
            flow_2d_preds[i] = padder.unpad(flow_2d_preds[i])

        for i in range(len(flow_3d_preds)):
            flow_3d_preds[i] = paral2persp(pc1 + flow_3d_preds[i], persp_cam_info, paral_cam_info) -\
                               paral2persp(pc1, persp_cam_info, paral_cam_info)

        # final_flow_2d 设置为 flow_2d_preds 最后一个元素的值。
        # 因此，如果在末尾添加新的预测结果，final_flow_2d 也会自动更新为新添加的元素。
        final_flow_2d = flow_2d_preds[-1]
        final_flow_3d = flow_3d_preds[-1]
        
        if 'flow_2d' not in inputs or 'flow_3d' not in inputs:
            return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d}

        target_2d = inputs['flow_2d'].float()
        target_3d = inputs['flow_3d'].float()
        
        # calculate losses
        loss_2d = calc_sequence_loss_2d(flow_2d_preds, target_2d, cfgs=self.cfgs.loss2d)
        loss_3d = calc_sequence_loss_3d(flow_3d_preds, target_3d, cfgs=self.cfgs.loss3d)
        # 总loss为光流和场景流loss的简单相加
        self.loss = loss_2d + loss_3d

        self.update_metrics('loss', self.loss)
        self.update_metrics('loss2d', loss_2d)
        self.update_metrics('loss3d', loss_3d)
        
        self.update_2d_metrics(final_flow_2d, target_2d)
        self.update_3d_metrics(final_flow_3d, target_3d)

        if 'occ_mask_3d' in inputs:
            self.update_3d_metrics(final_flow_3d, target_3d, inputs['occ_mask_3d'])


        return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d}

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        if best_metrics is None:
            return True
        return curr_metrics['epe2d'] < best_metrics['epe2d']
