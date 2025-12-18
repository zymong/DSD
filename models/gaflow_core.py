import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample, avg_pool2d
from mmdet.models.backbones import ResNet
from .utils import convex_upsample, mesh_grid, timer
from .mlp import Conv2dNormRelu
# from GAmodel import encoder_gmflownet
# from GAmodel import modules
# from encoder_gmflownet import BasicConvEncoder, POLAUpdate, MixAxialPOLAUpdate
# from modules import GCL, GGAM
from .encoder_gmflownet import BasicConvEncoder, POLAUpdate, MixAxialPOLAUpdate
from .modules import GCL, GGAM
from .gma import Attention, Aggregate

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

class Encoder2D(ResNet):
    def __init__(self, depth=50, pretrained=None):
        super().__init__(
            depth=depth,
            num_stages=2,
            strides=(1, 2),
            dilations=(1, 1),
            out_indices=(1,),
            norm_eval=True,
            with_cp=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=pretrained
            )
        )
        self.align = Conv2dNormRelu(self.feat_dim, 128)

        # MMCV, please shut up
        from mmcv.utils.logging import get_logger
        get_logger('root').setLevel(logging.ERROR)
        get_logger('mmcv').setLevel(logging.ERROR)

        self.init_weights()
    
    @timer.timer_func
    def forward(self, x):
        x = super().forward(x)[0]
        x = self.align(x)
        return x


class Correlation2D(nn.Module):
    def __init__(self, num_levels=4, radius=4):
        super().__init__()

        self.num_levels = num_levels
        self.radius = radius
        # self.fnet_aligner = nn.Conv2d(128, 256, kernel_size=1)
        
        # cost volume pyramid is built during runtime
        self.cost_volume_pyramid = None

    def build_cost_volume_pyramid(self, fmap1, fmap2):
        # fmap1 = self.fnet_aligner(fmap1.float())
        # fmap2 = self.fnet_aligner(fmap2.float())
        
        # all pairs correlation
        bs, dim, h, w = fmap1.shape
        fmap1 = fmap1.view(bs, dim, h * w)
        fmap2 = fmap2.view(bs, dim, h * w)

        cost_volume = torch.matmul(fmap1.transpose(1, 2), fmap2)
        cost_volume = cost_volume / torch.sqrt(torch.tensor(dim))
        cost_volume = cost_volume.reshape(bs * h * w, 1, h, w)

        self.cost_volume_pyramid = [cost_volume]
        for _ in range(self.num_levels-1):
            cost_volume = avg_pool2d(cost_volume, 2, stride=2)
            self.cost_volume_pyramid.append(cost_volume)

    @timer.timer_func
    def forward(self, coords):
        coords = coords.permute(0, 2, 3, 1).float()
        bs, h, w, _ = coords.shape
        r = self.radius

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.cost_volume_pyramid[i]
            # 使用torch.linspace函数创建一个在-r和r之间均匀分布的一维张量dx，表示水平方向上的偏移量。
            # 同样，创建一个在-r和r之间均匀分布的一维张量dy，表示垂直方向上的偏移量。
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            # 创建一个四维张量delta，表示相关性窗口中的所有偏移量的组合
            # 它的形状是(2*r+1, 2*r+1, 2)。
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)

            centroid_lvl = coords.reshape(bs*h*w, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # 双线性插值
            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(bs, h, w, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out
    
    @staticmethod
    def bilinear_sampler(feat, coords):
        """ Wrapper for grid_sample, uses pixel coordinates """
        h, w = feat.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (w - 1) - 1
        ygrid = 2 * ygrid / (h - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        feat = grid_sample(feat, grid, align_corners=True)

        return feat


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius, k_conv):
        super(SKMotionEncoder, self).__init__()
        corr_planes = corr_levels * (2 * corr_radius + 1) ** 2

        self.convc1 = PCBlock4_Deep_nopool_res(corr_planes, 256, k_conv=k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-2, k_conv=k_conv)

    @timer.timer_func
    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))

        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)




class ConvexUpsampler2D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
        )

    @timer.timer_func
    def forward(self, h, flow):
        # scale mask to balance gradients
        up_mask = 0.25 * self.mask(h.float())
        return convex_upsample(flow, up_mask)


class GAFlowCore(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        self.UpdateBlock = 'SKUpdateBlock6_Deep_nopoolres_AllDecoder'
        self.k_conv = [1, 15]
        self.PCUpdater_conv = [1, 7] 

        if 'dropout' not in self.cfgs:
            self.dropout = 0

        # if cfgs.dataset == 'sintel':
        #     self.fnet = nn.Sequential(
        #                     BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout),
        #                     MixAxialPOLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7)
        #                 )
        # else:
        #     self.fnet = nn.Sequential(
        #         BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout),
        #         POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        #     )
        

        # feature network, context network, and update block
        if self.cfgs.pola:
            self.fnet = nn.Sequential(
                BasicConvEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout),
                POLAUpdate(embed_dim=128, depth=6, num_head=8, window_size=7, neig_win_num=1)
            )
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.dropout)
        else:
            self.fnet = Encoder2D(cfgs.backbone.depth, cfgs.backbone.pretrained)
            self.cnet = Encoder2D(cfgs.backbone.depth, cfgs.backbone.pretrained)
        
        self.fnet_aligner = nn.Conv2d(128, 256, kernel_size=1)
        self.cnet_aligner = nn.Conv2d(128, 256, kernel_size=1)
        self.correlation = Correlation2D(self.corr_levels, self.corr_radius)
        
        self.motion_encoder = SKMotionEncoder(self.corr_levels, self.corr_radius, self.k_conv)
        # self.gru = PCBlock4_Deep_nopool_res(128+hdim+hdim+128, 128, k_conv=self.PCUpdater_conv)
        # GGAM flase
        self.gru = PCBlock4_Deep_nopool_res(128+hdim+hdim, 128, k_conv=self.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=self.k_conv)
        self.convex_upsampler = ConvexUpsampler2D(self.hidden_dim)
        self.gcl = GCL(embed_dim=256, depth=1, args=cfgs)
        if self.cfgs.ggam:
            self.gru = PCBlock4_Deep_nopool_res(128+hdim+hdim+128, 128, k_conv=self.PCUpdater_conv)
            self.aggregator = GGAM(args=None, chnn=128)
        if self.cfgs.gma:
            self.gru = PCBlock4_Deep_nopool_res(128+hdim+hdim+128, 128, k_conv=self.PCUpdater_conv)
            self.att = Attention(args=None, dim=cdim, heads=1, max_pos_size=160, dim_head=cdim)
            self.aggregator = Aggregate(args=None, dim=128, dim_head=128, heads=1)          
        # self.zero = nn.Parameter(torch.zeros(12))

    def forward(self, image1, image2):
        # run the feature network
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        gcl_fmap1 = self.gcl(fmap1)
        gcl_fmap2 = self.gcl(fmap2)

        # all-pair correlation
        self.correlation.build_cost_volume_pyramid(gcl_fmap1, gcl_fmap2)

        # run the context network
        cnet = self.cnet(image1)
        cnet = self.cnet_aligner(cnet)
        h, x = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        h = torch.tanh(h)
        x = torch.relu(x)
        attention = self.att(x)

        bs, _, image_h, image_w = image1.shape
        grid_coords = mesh_grid(bs, image_h//8, image_w//8, device=image1.device)

        '''
        softCorrMap = None
        if self.cfgs.dataset == 'sintel':
            # Correlation as initialization
            N, fC, fH, fW = fmap1.shape
            corr_gm = torch.einsum('b c n, b c m -> b n m', fmap1.view(N, fC, -1), fmap2.view(N, fC, -1))
            corr_gm = corr_gm / torch.sqrt(torch.tensor(fC).float())
            corrMap = corr_gm

            #_, coords_index = torch.max(corrMap, dim=-1) # no gradient here
            softCorrMap = F.softmax(corrMap, dim=2) * F.softmax(corrMap, dim=1) # (N, fH*fW, fH*fW)

        if self.cfgs.global_flow:
            # mutual match selection
            match12, match_idx12 = softCorrMap.max(dim=2) # (N, fH*fW)
            match21, match_idx21 = softCorrMap.max(dim=1)

            for b_idx in range(N):
                match21_b = match21[b_idx,:]
                match_idx12_b = match_idx12[b_idx,:]
                match21[b_idx,:] = match21_b[match_idx12_b]

            matched = (match12 - match21) == 0  # (N, fH*fW)
            coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(softCorrMap.device)
            coords_index[matched] = match_idx12[matched]

            # matched coords
            coords_index = coords_index.reshape(N, fH, fW)
            coords_x = coords_index % fW
            # coords_y = coords_index // fW
            coords_y = torch.div(coords_index, fW, rounding_mode='trunc')

            coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            coords1 = coords_xy
            '''

        flow_preds = []
        flow_pred = torch.zeros_like(grid_coords)

        if self.training:
            n_iters = self.cfgs.n_iters_train
        else:
            n_iters = self.cfgs.n_iters_eval

        for itr in range(n_iters):
            flow_pred = flow_pred.detach()
            
            # index correlation volume
            corr = self.correlation(grid_coords + flow_pred)

            # motion features: previous flow with current correlation 运动特征：先前的光流与当前的相关性
            motion_features = self.motion_encoder(flow_pred, corr) 

            # GGAM
            if self.cfgs.ggam:
                motion_features_global = self.aggregator(x, motion_features, itr)
                inp_cat = torch.cat([x, motion_features, motion_features_global], dim=1)
                # GRU
                h = self.gru(torch.cat([h, inp_cat], dim=1))
            # GMA
            elif self.cfgs.gma:
                motion_features_global = self.aggregator(attention, motion_features)
                inp_cat = torch.cat([x, motion_features, motion_features_global], dim=1)
                # GRU
                h = self.gru(torch.cat([h, inp_cat], dim=1))
            else:
                inp_cat = torch.cat([x, motion_features], dim=1)
                h = self.gru(torch.cat([h, inp_cat], dim=1))
            

            # predict delta flow
            delta_flow = self.flow_head(h)

            # F(t+1) = F(t) + \Delta(t)
            flow_pred = flow_pred + delta_flow

            # upsample predictions
            flow_up = self.convex_upsampler(h, flow_pred)

            flow_preds.append(flow_up)
        
        return flow_preds
