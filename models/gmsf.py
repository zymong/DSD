from .base import FlowModel
# from .gmsf_core import GMSF_Core
from .losses import calc_sequence_loss_3d, calc_sequence_loss_3d_addfeature
from .ids import persp2paral, paral2persp
import torch
import torch.nn as nn
import torch.nn.functional as F
from .point_conv import PointConvDW, PointConv
from .utils import backwarp_3d, batch_indexing, knn_interpolation, k_nearest_neighbor, build_pc_pyramid, timer
from .mlp import Conv1dNormRelu, MLP1d, MLP2d
from .transformer import FeatureTransformer3D, FeatureTransformer3D_PT
from .GMA3D import Gma3D
import copy
import numpy as np


class Encoder3D(nn.Module):
    def __init__(self, n_channels, norm=None, k=16):
        super().__init__()

        self.level0_mlp = MLP1d(3, [n_channels[0], n_channels[0]])

        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(len(n_channels) - 1):
            self.mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.convs.append(PointConv(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    @timer.timer_func
    def forward(self, xyzs):
        """
        :param xyzs: pyramid of points
        :return feats: pyramid of features
        """
        assert len(xyzs) == len(self.mlps) + 1

        inputs = xyzs[0]  # [bs, 3, n_points]
        feats = [self.level0_mlp(inputs)]

        for i in range(len(xyzs) - 1):
            feat = self.mlps[i](feats[-1])
            feat = self.convs[i](xyzs[i], feat, xyzs[i + 1])
            feats.append(feat)

        return feats


class Correlation3D(nn.Module):
    def __init__(self, out_channels, k=16):
        super().__init__()
        self.k = k

        self.cost_mlp = MLP2d(4, [out_channels // 4, out_channels // 4], act='relu')
        self.merge = Conv1dNormRelu(out_channels, out_channels)  # ?

        # cost volume is built during runtime
        self.cost_volume_pyramid = None

    def build_cost_volume_pyramid(self, feat1, feat2, xyzs2, k=3):
        cost_volume = torch.bmm(feat1.float().transpose(1, 2), feat2.float())  # [B, N, N]
        cost_volume = cost_volume / feat1.shape[1]
        self.cost_volume_pyramid = [cost_volume]  # [B, N, M0]

        for i in range(1, len(xyzs2)):
            knn_indices = k_nearest_neighbor(xyzs2[i - 1], xyzs2[i], k=k)
            knn_corr = batch_indexing(self.cost_volume_pyramid[i - 1], knn_indices)
            avg_corr = torch.mean(knn_corr, dim=-1)
            self.cost_volume_pyramid.append(avg_corr)

    def calc_matching_cost(self, xyz1, xyz2, cost_volume):
        bs, n_points1, n_points2 = cost_volume.shape

        # for each point in xyz1, find its neighbors in xyz2
        knn_indices_cross = k_nearest_neighbor(input_xyz=xyz2, query_xyz=xyz1, k=self.k)  # [bs, n_points, k]
        knn_xyz2 = batch_indexing(xyz2, knn_indices_cross)
        knn_xyz2_norm = knn_xyz2 - xyz1.view(bs, 3, n_points1, 1)
        
        knn_corr = batch_indexing(
            cost_volume.reshape(bs * n_points1, n_points2),
            knn_indices_cross.reshape(bs * n_points1, self.k),
            layout='channel_last'
        ).reshape(bs, 1, n_points1, self.k)

        cost = self.cost_mlp(torch.cat([knn_xyz2_norm, knn_corr], dim=1))
        cost = torch.sum(cost, dim=-1)

        return cost

    @timer.timer_func
    def forward(self, xyz1, xyzs2):
        """
        :param xyz1: [batch_size, 3, n_points]
        :param feat1: [batch_size, in_channels, n_points]
        :param xyz2: [batch_size, 3, n_points]
        :param feat2: [batch_size, in_channels, n_points]
        :param knn_indices_1in1: for each point in xyz1, find its neighbors in xyz1, [batch_size, n_points, k]
        :return cost volume for each point in xyz1: [batch_size, n_cost_channels, n_points]
        """
        # compute single-scale matching cost
        cost0 = self.calc_matching_cost(xyz1, xyzs2[0], self.cost_volume_pyramid[0])
        cost1 = self.calc_matching_cost(xyz1, xyzs2[1], self.cost_volume_pyramid[1])
        cost2 = self.calc_matching_cost(xyz1, xyzs2[2], self.cost_volume_pyramid[2])
        cost3 = self.calc_matching_cost(xyz1, xyzs2[3], self.cost_volume_pyramid[3])

        # merge multi-scale costs
        costs = torch.cat([cost0, cost1, cost2, cost3], dim=1)
        costs = self.merge(costs)

        return costs


class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.conv1 = PointConvDW(input_dim, 128, k=32)  # ?
        self.conv2 = PointConvDW(128, 64, k=32)  # ?
        self.fc = nn.Conv1d(64, 3, kernel_size=1)

    @timer.timer_func
    def forward(self, xyz, features, knn_indices=None):
        features = features.float()
        features = self.conv1(xyz, features, knn_indices=knn_indices)
        features = self.conv2(xyz, features, knn_indices=knn_indices)
        return self.fc(features)


class GRU3D(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv_z = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)
        self.conv_r = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)
        self.conv_q = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)

    @timer.timer_func
    def forward(self, xyz, h, x, knn_indices=None):    
        h, x = h.float(), x.float()
        hx = torch.cat([h, x], dim=1)
       
        z = torch.sigmoid(self.conv_z(xyz, hx, knn_indices=knn_indices))
        r = torch.sigmoid(self.conv_r(xyz, hx, knn_indices=knn_indices))
        q = torch.tanh(self.conv_q(xyz, torch.cat([r * h, x], dim=1), knn_indices=knn_indices))
        h = (1 - z) * h + z * q
        return h


class MotionEncoder3D(nn.Module):
    def __init__(self, corr_dim=128):
        super(MotionEncoder3D, self).__init__()
        self.conv_c1 = PointConvDW(corr_dim, corr_dim)
        self.conv_f1 = PointConvDW(3, 32, k=32)
        self.conv_f2 = PointConvDW(32, 16, k=16)
        self.conv = PointConvDW(corr_dim + 16, 128 - 3, k=16)  # ?

    @timer.timer_func
    def forward(self, xyz, flow, corr, knn_indices):
        corr, flow = corr.float(), flow.float()
        corr_feat = self.conv_c1(xyz, corr, knn_indices=knn_indices)
        flow_feat = self.conv_f1(xyz, flow, knn_indices=knn_indices)
        flow_feat = self.conv_f2(xyz, flow_feat, knn_indices=knn_indices)

        corr_flow_feat = torch.cat([corr_feat, flow_feat], dim=1)
        out = self.conv(xyz, corr_flow_feat, knn_indices=knn_indices)

        return torch.cat([out, flow], dim=1)
    

def global_correlation_softmax_3d(feature0, feature1, xyzs1, xyzs2,
                               ):
    # global correlation
    b, c, n = feature0.shape
    feature0 = feature0.permute(0, 2, 1)  # [B, N, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, N]

    correlation = torch.matmul(feature0, feature1).view(b, n, n) / (c ** 0.5)  # [B, N, N]

    # flow from softmax
    init_grid_1 = xyzs1.to(correlation.device) # [B, 3, N]
    init_grid_2 = xyzs2.to(correlation.device) # [B, 3, N]
    grid_2 = init_grid_2.permute(0, 2, 1)  # [B, N, 3]

    correlation = correlation.view(b, n, n)  # [B, N, N]

    prob = F.softmax(correlation, dim=-1)  # [B, N, N]

    correspondence = torch.matmul(prob, grid_2).view(b, n, 3).permute(0, 2, 1)  # [B, 3, N]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid_1

    return flow, prob

class SelfCorrelationSoftmax3D(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(SelfCorrelationSoftmax3D, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                **kwargs,
                ):
        # q, k: feature [B, C, N], v: flow [B, 3, N]

        b, c, n = feature0.size()

        query = feature0.permute(0, 2, 1)  # [B, N, C]

        query = self.q_proj(query)  # [B, N, C]
        key = self.k_proj(query)  # [B, N, C]

        value = flow.view(b, flow.size(1), n).permute(0, 2, 1)  # [B, N, 3]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, N, N]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, N, 3]
        out = out.view(b, n, value.size(-1)).permute(0, 2, 1)  # [B, 3, N]

        return out

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=32, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1) # (BNk)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :] # (BNk C)
    feature = feature.view(batch_size, num_points, k, num_dims) # (B N k C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # (B N k C)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (B 2C N k)
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, output_channels=128, k=32): # 32
        super(DGCNN, self).__init__()
        self.outchannel = output_channels
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(self.outchannel)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(320, self.outchannel, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x): # x torch.Size([B, 3, 8192])
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) # (B 2C N k)
        x = self.conv1(x) # (B 64 N k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (B 64 N)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        return x



class GMSF(FlowModel):
    def __init__(self, cfgs):
        super(GMSF, self).__init__()
        self.cfgs = cfgs
        # self.core = GMSF_Core(cfgs)
        self.feature_channels = feature_channels = 128
        self.num_transformer_layers = num_transformer_layers = 10
        self.num_transformer_pt_layers = num_transformer_pt_layers = 1
        self.ffn_dim_expansion = ffn_dim_expansion = 4


        # Transformer
        if self.num_transformer_layers > 0:
            self.transformer = FeatureTransformer3D(num_layers=num_transformer_layers,
                                                d_model=feature_channels,
                                                ffn_dim_expansion=ffn_dim_expansion,
                                                )
        if self.num_transformer_pt_layers > 0:
            self.transformer_PT = FeatureTransformer3D_PT(num_layers=num_transformer_pt_layers,
                                                        d_points=feature_channels,
                                                        ffn_dim_expansion=ffn_dim_expansion,
                                                        )

        # self correlation with self-feature similarity
        if self.cfgs.refinement:
            self.feature_flow_attn = SelfCorrelationSoftmax3D(in_channels=feature_channels)

        if self.cfgs.DGCNN:
            self.DGCNN = DGCNN(output_channels=self.feature_channels, k=32)
        else:
            self.fnet = Encoder3D(n_channels=[64, 96, 128], norm='batch_norm', k=16)
            self.cnet = Encoder3D(n_channels=[64, 96, 128], norm='batch_norm', k=16)
        self.cnet_aligner = nn.Conv1d(128, 256, kernel_size=1)
        self.correlation = Correlation3D(out_channels=128, k=32)
        self.motion_encoder = MotionEncoder3D(corr_dim=128)
        self.gru = GRU3D(input_dim=128 + 128, hidden_dim=128)
        self.flow_head = FlowHead3D(input_dim=128)
        if self.cfgs.gma:
            self.gru = GRU3D(input_dim=128 + 128 + 128, hidden_dim=128)
            self.gma = Gma3D(gma_dim=128)
        if self.cfgs.feature_migration:
            self.DGCNN2 = DGCNN(output_channels=self.feature_channels, k=16)
            # self.correlation2 = Correlation3D(out_channels=128, k=16)
            self.transformer2 = FeatureTransformer3D(num_layers=num_transformer_layers,
                                                d_model=feature_channels,
                                                ffn_dim_expansion=ffn_dim_expansion,
                                                )
            self.transformer_PT2 = FeatureTransformer3D_PT(num_layers=num_transformer_pt_layers,
                                                        d_points=feature_channels,
                                                        ffn_dim_expansion=ffn_dim_expansion,
                                                        )


    def forward(self, inputs):
        pc1, pc2 = inputs['pcs'][:, :3], inputs['pcs'][:, 3:]
        intrinsics = inputs['intrinsics']


        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': 540,
            'sensor_w': 960,
            'f': intrinsics[:, 0],
            'cx': intrinsics[:, 1],
            'cy': intrinsics[:, 2],
        }

        if self.cfgs.ids.enabled:
            paral_cam_info = {
                'projection_mode': 'parallel',
                'sensor_h': round(540 / 32),
                'sensor_w': round(960 / 32),
                'cx': (round(960 / 32) - 1) / 2,
                'cy': (round(540 / 32) - 1) / 2,
            }
            pc1 = persp2paral(pc1, persp_cam_info, paral_cam_info)
            pc2 = persp2paral(pc2, persp_cam_info, paral_cam_info)
        else:
            paral_cam_info = None

        flow_preds = []

        xyzs1, xyzs2, _, _ = build_pc_pyramid(
            pc1, pc2, [4096, 2048, 1024]
            # pc1, pc2, [2048, 1024, 512]
            # pc1, pc2, [1024, 512, 256]
            # pc1, pc2, [512, 256, 128]
        )
        # print(xyzs1[0].shape)
        # print(xyzs2[0].shape)

        if self.cfgs.DGCNN:
            feat1 = self.DGCNN(xyzs1[0])    
            feat2 = self.DGCNN(xyzs2[0])
            # featc = self.DGCNN(xyzs1[0])
        else:
            feat1 = self.fnet(xyzs1[:3])[2]
            feat2 = self.fnet(xyzs2[:3])[2]
            # featc = self.cnet(xyzs1[:3])[2]
        

        xyzs1, xyzs2 = xyzs1[0:], xyzs2[0:]
        xyz1, xyz2 = xyzs1[0], xyzs2[0]  

        # Transformer
        if self.num_transformer_pt_layers > 0:
            feat1, feat2 = self.transformer_PT(xyz1, xyz2, 
                                                    feat1, feat2,
                                                    )

        if self.num_transformer_layers > 0:
            feat1, feat2 = self.transformer(feat1, feat2,
                                                )

        if self.cfgs.feature_migration:
        # if False:
            pc1_8192 = inputs['pcs_8192'][:, :3]
            pc2_8192 = inputs['pcs_8192'][:, 3:]
            feat1_8192 = self.DGCNN(pc1_8192)    
            feat2_8192 = self.DGCNN(pc2_8192)
            feat1_8192, feat2_8192 = self.transformer_PT(pc1_8192, pc2_8192, feat1_8192, feat2_8192)
            feat1_8192, feat2_8192 = self.transformer(feat1_8192, feat2_8192)
            indices1 = inputs['indices1']
            indices2 = inputs['indices2']
            
            indices1 = indices1.unsqueeze(1)  
            indices2 = indices2.unsqueeze(1) 
            
            feat1_good = torch.gather(feat1_8192, 2, indices1.expand(-1, feat1_8192.size(1), -1))  # [batch_size, channels, k]
            feat2_good = torch.gather(feat2_8192, 2, indices2.expand(-1, feat2_8192.size(1), -1))  # [batch_size, channels, k]
           
            feat1 = feat1_good
            feat2 = feat2_good
            
        
        self.correlation.build_cost_volume_pyramid(feat1, feat2, xyzs2)

        if self.cfgs.ag_featc:
            # featc = copy.deepcopy(feat1)  # 创建feat1的深拷贝
            featc = feat1.detach().clone()

        featc = self.cnet_aligner(featc)

        # h:hidden features
        # x:context_features
        h, x = torch.split(featc, [128, 128], dim=1)
        h = torch.tanh(h)
        x = torch.relu(x)
        
        knn_indices = k_nearest_neighbor(xyz1, xyz1, k=32)

        if self.training:
            n_iters = self.cfgs.n_iters_train
        else:
            n_iters = self.cfgs.n_iters_eval

        for it in range(n_iters):
            if it > 0:
                flow_pred = flow_pred.detach()
                xyzs2_warp = [backwarp_3d(xyz1, xyz2_lvl, flow_pred) for xyz2_lvl in xyzs2]
            else:
                flow_pred = torch.zeros_like(xyz1)
                xyzs2_warp = xyzs2

            # correlation
            corr = self.correlation(xyz1, xyzs2_warp)

            # motion feat: corr + flow
            motion_feat = self.motion_encoder(xyz1, flow_pred, corr, knn_indices=knn_indices)

            # GMA3D
            if self.cfgs.gma:
                gma = self.gma(h, motion_feat, xyz1)
                x_gma = torch.cat([x, motion_feat, gma], dim=1)
                # GRU
                h = self.gru(xyz1, h=h, x=x_gma, knn_indices=knn_indices)

            else:
                h = self.gru(xyz1, h=h, x=torch.cat([x, motion_feat], dim=1), knn_indices=knn_indices)
            # predict delta flow
            flow_delta = self.flow_head(xyz1, h, knn_indices)
            flow_pred = flow_pred + flow_delta.float()

            # flow refinement with self-attn
            if self.cfgs.refinement:
                flow_pred = self.feature_flow_attn(feat1, flow_pred)
            
            flow_preds.append(flow_pred)
        
        # for i in range(len(flow_preds)):
        #     flow_preds[i] = knn_interpolation(xyz1, flow_preds[i], pc1, k=3)

        if self.cfgs.ids.enabled:
            for i in range(len(flow_preds)):
                flow_preds[i] = paral2persp(pc1 + flow_preds[i], persp_cam_info, paral_cam_info) - \
                                paral2persp(pc1, persp_cam_info, paral_cam_info)

        final_flow_3d = flow_preds[-1]
        # flow refinement with self-attn
        # if self.cfgs.refinement:
        #     final_flow_3d = self.feature_flow_attn(feat1, final_flow_3d)
        # print(final_flow_3d.shape) 8192

        target_3d = inputs['flow_3d'][:, :3]
        self.loss = calc_sequence_loss_3d(flow_preds, target_3d, self.cfgs.loss)

        # prepare scalar summary
        self.update_metrics('loss3d', self.loss)

        if self.cfgs.feature_migration:
            self.loss = calc_sequence_loss_3d_addfeature(flow_preds, target_3d, feat1_good, feat2_good, feat1, feat2, self.cfgs.loss)
        else:
            self.loss = calc_sequence_loss_3d(flow_preds, target_3d, self.cfgs.loss)
        self.update_3d_metrics(final_flow_3d, target_3d)

        return {'flow_3d': final_flow_3d}

    @staticmethod
    def is_better(curr_summary, best_summary):
        if best_summary is None:
            return True
        return curr_summary['epe3d'] < best_summary['epe3d']
