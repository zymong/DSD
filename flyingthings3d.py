import os
import cv2
import numpy as np
import torch.utils.data
from utils import load_flow_png
from augmentation import joint_augmentation, joint_augmentation_pc
import glob


class FlyingThings3D(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.cfgs = cfgs

        self.indices = []
        for filename in os.listdir(os.path.join(self.root_dir, self.split, 'flow_2d')):
            self.indices.append(int(filename.split('.')[0]))
        self.indices = sorted(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(0)

        idx1 = self.indices[i]
        idx2 = idx1 + 1
        data_dict = {'index': idx1}

        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        # load 2D data
        if self.cfgs.pass_name == 'cleanfinal' and self.cfgs.augmentation.enabled:
            pass_name = 'clean' if np.random.randint(2) == 0 else 'final'
        else:
            pass_name = self.cfgs.pass_name

        image1 = cv2.imread(os.path.join(self.split_dir, 'image_%s' % pass_name, '%07d.png' % idx1))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.split_dir, 'image_%s' % pass_name, '%07d.png' % idx2))[..., ::-1]
        flow_2d, flow_mask_2d = load_flow_png(os.path.join(self.split_dir, 'flow_2d', '%07d.png' % idx1))

        # load 3D data
        pc_dict = np.load(os.path.join(self.split_dir, 'pc', '%07d.npz' % idx1))
        flow_3d = np.load(os.path.join(self.split_dir, 'flow_3d', '%07d.npy' % idx1))
        pc1, pc2 = pc_dict['pc1'], pc_dict['pc2']

        # load occlusion mask (only for evaluation)
        if os.path.exists(os.path.join(self.split_dir, 'occ_mask_3d')):
            occ_mask_3d = np.load(os.path.join(self.split_dir, 'occ_mask_3d', '%07d.npy' % idx1))
            occ_mask_3d = np.unpackbits(occ_mask_3d, count=len(pc1))
        else:
            occ_mask_3d = np.zeros(len(pc1), dtype=np.bool)

        # ignore fast moving objects
        flow_mask_2d = np.logical_and(flow_mask_2d, np.linalg.norm(flow_2d, axis=-1) < 250.0)
        flow_2d = np.concatenate([flow_2d, flow_mask_2d[..., None].astype(np.float32)], axis=2)

        # data augmentation
        while True:
            try:
                results = joint_augmentation(
                    image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation
                )
            except AssertionError:
                continue
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = results
            break

        # if self.cfgs.n_points < 8192:
        #     indices1_8192 = np.random.choice(pc1.shape[0], size=8192, replace=pc1.shape[0] < self.cfgs.n_points)
        #     indices2_8192 = np.random.choice(pc2.shape[0], size=8192, replace=pc2.shape[0] < self.cfgs.n_points)
        #     pc1, pc2, flow_3d, occ_mask_3d = pc1[indices1_8192], pc2[indices2_8192], flow_3d[indices1_8192], occ_mask_3d[indices1_8192]
        #     data_dict['pcs_8192'] = np.concatenate([pc1, pc2], axis=1).transpose()

        if self.cfgs.augmentation.enabled or pc1.shape[0] != self.cfgs.n_points:
            indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
            indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
            pc1, pc2, flow_3d, occ_mask_3d = pc1[indices1], pc2[indices2], flow_3d[indices1], occ_mask_3d[indices1]
            data_dict['indices1'] = indices1
            data_dict['indices2'] = indices2

        if self.cfgs.with_pc:
            pc_pair = np.concatenate([pc1, pc2], axis=1)
            data_dict['pcs'] = pc_pair.transpose()
            data_dict['flow_3d'] = flow_3d.transpose()
            data_dict['intrinsics'] = np.float32([f, cx, cy])
            data_dict['occ_mask_3d'] = occ_mask_3d

            
    
        if self.cfgs.with_image:
            image_pair = np.concatenate([image1, image2], axis=-1)
            data_dict['images'] = image_pair.transpose([2, 0, 1])
            data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])

        return data_dict


class FlyingThings3D_hpl(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.root_dir = str(cfgs.root_dir)
        # print(root_dir)   
        self.split = str(cfgs.split)
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.n_points = cfgs.n_points # 8192
        self.cfgs = cfgs


        self.indices = []
        for filename in os.listdir(os.path.join(self.root_dir, self.split, 'flow_2d')):
            self.indices.append(int(filename.split('.')[0]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        augmentation = True
        if not augmentation:
            np.random.seed(0)

        idx1 = self.indices[i]
        idx2 = idx1 + 1
        data_dict = {'index': idx1}
        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        # load data
        pcs = np.load(os.path.join(self.split_dir, 'pc', '%07d.npz' % idx1))
        pc1, pc2 = pcs['pc1'], pcs['pc2']

        flow_2d, flow_mask_2d = load_flow_png(os.path.join(self.split_dir, 'flow_2d', '%07d.png' % idx1))
        flow_3d = np.load(os.path.join(self.split_dir, 'flow_3d', '%07d.npy' % idx1))

        occ_mask_3d = np.load(os.path.join(self.split_dir, 'occ_mask_3d', '%07d.npy' % idx1))
        occ_mask_3d = np.unpackbits(occ_mask_3d, count=len(pc1))

        # ignore fast moving objects
        flow_mask_2d = np.logical_and(flow_mask_2d, np.linalg.norm(flow_2d, axis=-1) < 250.0)
        flow_2d = np.concatenate([flow_2d, flow_mask_2d[..., None].astype(np.float32)], axis=2)

        if self.split == 'train':
            pc1, pc2, flow_3d, f, cx, cy = joint_augmentation_pc(
                pc1, pc2, flow_3d, f, cx, cy, image_h=540, image_w=960
            )

        # random sampling during training
        if self.split == 'train':
            indices1 = np.random.choice(pc1.shape[0], size=self.n_points, replace=pc1.shape[0] < self.n_points)
            indices2 = np.random.choice(pc2.shape[0], size=self.n_points, replace=pc2.shape[0] < self.n_points)       
            pc1, pc2, flow_3d, occ_mask_3d = pc1[indices1], pc2[indices2], flow_3d[indices1], occ_mask_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1)

        # flow_2d = torch.from_numpy(flow_2d).permute(2, 0, 1).float()
        # pcs = torch.from_numpy(pcs).float()
        # flow_3d = torch.from_numpy(flow_3d).permute(1, 0).float()
        # intrinsics = torch.from_numpy(np.float32([f, cx, cy]))
        # occ_mask_3d = torch.from_numpy(occ_mask_3d)

        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['occ_mask_3d'] = occ_mask_3d
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict


class FlyingThings3DSubset_FlowNet3D(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        self.root_dir = cfgs.root_dir
        self.n_points = cfgs.n_points
        self.train = cfgs.train
        # print(f"Dataset root: {self.root_dir}")  # Debugging info
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root_dir, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root_dir, 'TEST*.npz'))
        # if not self.datapath:
        #     raise FileNotFoundError(f"No TRAIN*.npz files found in {self.root}")
        # print(f"Data paths: {self.datapath}")  # Debugging info
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        data_dict = {'index': index}
        if index in self.cache:
            # pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
            pos1, pos2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype(np.float32)
                pos2 = data['points2'].astype(np.float32)
                # color1 = data['color1'] / 255
                # color2 = data['color2'] / 255
                flow = data['flow'].astype(np.float32)
                mask1 = data['valid_mask1'].astype(bool)

            if len(self.cache) < self.cache_size:
                # self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)
                self.cache[index] = (pos1, pos2, flow, mask1)
        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        if self.train:
            pos1, pos2, flow, f, cx, cy = joint_augmentation_pc(
                pos1, pos2, flow, f, cx, cy, image_h=540, image_w=960
            )
        
        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.n_points, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.n_points, replace=False)

            pos1_ = np.copy(pos1[sample_idx1, :])
            pos2_ = np.copy(pos2[sample_idx2, :])
            # color1_ = np.copy(color1[sample_idx1, :])
            # color2_ = np.copy(color2[sample_idx2, :])
            flow_ = np.copy(flow[sample_idx1, :])
            mask1_ = np.copy(mask1[sample_idx1])
        else:
            pos1_ = np.copy(pos1[:self.n_points, :])
            pos2_ = np.copy(pos2[:self.n_points, :])
            # color1_ = np.copy(color1[:self.n_points, :])
            # color2_ = np.copy(color2[:self.n_points, :])
            flow_ = np.copy(flow[:self.n_points, :])
            mask1_ = np.copy(mask1[:self.n_points])

        pcs = np.concatenate([pos1_, pos2_], axis=1)
        flow_3d = flow_
        # pcs = torch.from_numpy(pcs).float()
        # flow_3d = torch.from_numpy(flow_).permute(1, 0).float()
        # intrinsics = torch.from_numpy(np.float32([f, cx, cy]))
        occ_mask_3d = (~mask1_)
        # occ_mask_3d = torch.from_numpy(~mask1_)

        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['occ_mask_3d'] = occ_mask_3d
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict

