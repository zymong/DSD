import os
import cv2
import numpy as np
import torch.utils.data
from utils import disp2pc, project_pc2image, load_flow_png, load_disp_png, load_calib, zero_padding
from augmentation import joint_augmentation

import os.path as osp

class ProcessData(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2 = data
        if pc1 is None:
            return None, None, None,

        sf = pc2[:, :3] - pc1[:, :3]

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)
        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc1, pc2, sf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string


class KITTI(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)
        assert cfgs.split in ['training200', 'training160', 'training40', 'testing200']

        if 'training' in cfgs.split:
            self.root_dir = os.path.join(cfgs.root_dir, 'training')
        else:
            self.root_dir = os.path.join(cfgs.root_dir, 'testing')

        self.split = cfgs.split
        self.cfgs = cfgs

        if self.split == 'training200' or self.split == 'testing200':
            self.indices = np.arange(200)
        elif self.split == 'training160':
            self.indices = [i for i in range(200) if i % 5 != 0]
        elif self.split == 'training40':
            self.indices = [i for i in range(200) if i % 5 == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(23333)

        index = self.indices[i]
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        image1 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_10.png' % index))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_11.png' % index))[..., ::-1]
        data_dict['input_h'] = image1.shape[0]
        data_dict['input_w'] = image1.shape[1]

        flow_2d, flow_2d_mask = load_flow_png(os.path.join(self.root_dir, 'flow_occ', '%06d_10.png' % index))

        disp1, mask1 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_0', '%06d_10.png' % index))
        disp2, mask2 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_1', '%06d_10.png' % index))
        mask = np.logical_and(np.logical_and(mask1, mask2), flow_2d_mask)

        pc1 = disp2pc(disp1, baseline=0.54, f=f, cx=cx, cy=cy)[mask]
        pc2 = disp2pc(disp2, baseline=0.54, f=f, cx=cx, cy=cy, flow=flow_2d)[mask]
        flow_3d = pc2 - pc1
        flow_3d_mask = np.ones(flow_3d.shape[0], dtype=np.float32)

        # remove out-of-boundary regions of pc2 to create occlusion
        image_h, image_w = disp2.shape[:2]
        xy2 = project_pc2image(pc2, image_h, image_w, f, cx, cy, clip=False)
        boundary_mask = np.logical_and(
            np.logical_and(xy2[..., 0] >= 0, xy2[..., 0] < image_w),
            np.logical_and(xy2[..., 1] >= 0, xy2[..., 1] < image_h)
        )
        pc2 = pc2[boundary_mask]

        flow_2d = np.concatenate([flow_2d, flow_2d_mask[..., None].astype(np.float32)], axis=-1)
        flow_3d = np.concatenate([flow_3d, flow_3d_mask[..., None].astype(np.float32)], axis=-1)

        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        padding_h, padding_w = 376, 1242
        image1 = zero_padding(image1, padding_h, padding_w)
        image2 = zero_padding(image2, padding_h, padding_w)
        flow_2d = zero_padding(flow_2d, padding_h, padding_w)

        # data augmentation
        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = joint_augmentation(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation,
        )

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1)
        images = np.concatenate([image1, image2], axis=-1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict


class KITTITest(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)
        assert cfgs.split in ['testing200']

        self.root_dir = os.path.join(cfgs.root_dir, 'testing')
        self.split = cfgs.split
        self.cfgs = cfgs

    def __len__(self):
        return 200

    def __getitem__(self, index):
        np.random.seed(23333)
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        image1 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_10.png' % index))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_11.png' % index))[..., ::-1]
        data_dict['input_h'] = image1.shape[0]
        data_dict['input_w'] = image1.shape[1]

        disp1, mask1 = load_disp_png(os.path.join(self.root_dir, 'disp_%s' % self.cfgs.disp_provider, '%06d_10.png' % index))
        disp2, mask2 = load_disp_png(os.path.join(self.root_dir, 'disp_%s' % self.cfgs.disp_provider, '%06d_11.png' % index))

        # ignore top 110 rows without evaluation
        mask1[:110] = 0
        mask2[:110] = 0

        pc1 = disp2pc(disp1, baseline=0.54, f=f, cx=cx, cy=cy)[mask1]
        pc2 = disp2pc(disp2, baseline=0.54, f=f, cx=cx, cy=cy)[mask2]

        # limit max height (2.0m)
        pc1 = pc1[pc1[..., 1] > -2.0]
        pc2 = pc2[pc2[..., 1] > -2.0]

        # limit max depth
        pc1 = pc1[pc1[..., -1] < self.cfgs.max_depth]
        pc2 = pc2[pc2[..., -1] < self.cfgs.max_depth]

        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        padding_h, padding_w = 376, 1242
        image1 = zero_padding(image1, padding_h, padding_w)
        image2 = zero_padding(image2, padding_h, padding_w)

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2 = pc1[indices1], pc2[indices2]

        pcs = np.concatenate([pc1, pc2], axis=1)
        images = np.concatenate([image1, image2], axis=-1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict
    


class KITTI_hplflownet(torch.utils.data.Dataset): # hplflownet
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 transform=None,
                 train=False,
                 num_points=8192,
                 data_root='/mnt/save_home_all/home/wxy/code/GMSF/datasets/KITTIs/',
                 remove_ground = True):
        # self.root = osp.join(data_root, 'KITTI_processed_occ_final')
        self.root = data_root
        #assert train is False
        self.train = train
        self.transform = ProcessData(data_process_args = {'DEPTH_THRESHOLD':35., 'NO_CORR':True},
                                    num_points=8192,
                                    allow_less_points=False)
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join('/mnt/kitti_scene_flow/training/', 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pcs = np.concatenate([pc1_transformed, pc2_transformed], axis=1)

        pcs = torch.from_numpy(pcs).permute(0, 1).float()
        flow_3d = torch.from_numpy(sf_transformed).permute(1, 0).float()
        intrinsics = torch.from_numpy(np.float32([f, cx, cy]))

        data_dict['pcs'] = pcs.transpose(0, 1)
        data_dict['flow_3d'] = flow_3d
        data_dict['intrinsics'] = intrinsics
        return data_dict

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy'))  #.astype(np.float32)

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]

        return pc1, pc2

class KITTI_flownet3d(torch.utils.data.Dataset): # flownet3d
    def __init__(self, split='training150', root='/mnt/kitti_scene_flow'):
        assert os.path.isdir(root)
        assert split in ['training200', 'training160', 'training150', 'training40']

        self.root_dir = os.path.join(root, 'training')
        self.split = split
        self.augmentation = False
        self.n_points = 8192# 8192
        self.max_depth = 30

        if self.split == 'training200':
            self.indices = np.arange(200)
        if self.split == 'training150':
            self.indices = np.arange(150)
        elif self.split == 'training160':
            self.indices = [i for i in range(200) if i % 5 != 0]
        elif self.split == 'training40':
            self.indices = [i for i in range(200) if i % 5 == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.augmentation:
            np.random.seed(23333)

        index = self.indices[i]
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]


        # path = os.path.join('datasets/KITTIo/', '%06d.npz' % index)
        path = os.path.join('/mnt/save_home_all/home/wxy/code/GMSF/datasets/KITTIo/', '%06d.npz' % index)
        data = np.load(path)
        pc1 = np.concatenate((data['pos1'][:,1:2], data['pos1'][:,2:3], data['pos1'][:,0:1]), axis=1)
        pc2 = np.concatenate((data['pos2'][:,1:2], data['pos2'][:,2:3], data['pos2'][:,0:1]), axis=1)
        # limit max depth
        #pc1 = pc1[pc1[..., -1] < self.max_depth]
        #pc2 = pc2[pc2[..., -1] < self.max_depth]
        flow_3d = np.concatenate((data['gt'][:,1:2], data['gt'][:,2:3], data['gt'][:,0:1]), axis=1)

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.n_points, replace=pc1.shape[0] < self.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.n_points, replace=pc2.shape[0] < self.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1)

        pcs = torch.from_numpy(pcs).permute(0, 1).float()
        flow_3d = torch.from_numpy(flow_3d).permute(1, 0).float()
        intrinsics = torch.from_numpy(np.float32([f, cx, cy]))

        # print("pcs shape", pcs.shape)
        # print("flow_3d shape", flow_3d.shape)
        data_dict['pcs'] = pcs.transpose(0, 1)
        data_dict['flow_3d'] = flow_3d
        data_dict['intrinsics'] = intrinsics

        return data_dict
