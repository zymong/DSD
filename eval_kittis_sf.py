# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import utils
# import hydra
# import shutil
# import logging
# import torch
# import torch.optim
# import torch.utils.data as data
# import torch.backends.cudnn as cudnn
# import numpy as np
# from tqdm import tqdm
# from omegaconf import DictConfig
# from factory import model_factory, dataset_factory
# from utils import copy_to_device, size_of_batch


# # python eval_things_noc_sf.py testset=flyingthings3d_subset_hpl model=camliga_l ckpt.path=outputs/camliga_l/2024-05-16/17-35-25/ckpts/epoch-100.pt
# # python eval_things_noc_sf.py testset=flyingthings3d_subset_hpl model=camliraft_l ckpt.path=outputs/camliraft_l/2024-05-09/16-37-26/ckpts/epoch-100.pt
# # python eval_things_noc_sf.py testset=ft3ds model=gmsf ckpt.path=/home/wxy/code/camliga/outputs/gmsf/2024-12-17/13-15-06/ckpts/best.pt


# class Evaluator:
#     def __init__(self, device: torch.device, cfgs: DictConfig):
#         self.cfgs = cfgs
#         self.device = device

#         logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
#         self.test_dataset = dataset_factory(self.cfgs.testset)
#         self.test_loader = utils.FastDataLoader(
#             dataset=self.test_dataset,
#             batch_size=self.cfgs.testset.batch_size,
#             num_workers=self.cfgs.testset.n_workers
#         )

#         logging.info('Creating model: %s' % self.cfgs.model.name)
#         self.model = model_factory(self.cfgs.model).to(device=self.device)
#         self.model.eval()

#         logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
#         checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
#         self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

#     @torch.no_grad()
#     def run(self):
#         logging.info('Running evaluation...')
#         metrics_3d = {'counts': 0, 'EPE3d': 0.0, 'AccS': 0.0, 'AccR': 0.0, 'Outlier': 0.0}

#         for inputs in tqdm(self.test_loader):
#             inputs = copy_to_device(inputs, self.device)

#             with torch.cuda.amp.autocast(enabled=False):
#                 outputs = self.model.forward(inputs)

#             for batch_id in range(size_of_batch(inputs)):
#                 flow_3d_pred = outputs['flow_3d'][batch_id]
#                 flow_3d_target = inputs['flow_3d'][batch_id]

#                 epe3d_map = torch.sqrt(torch.sum((flow_3d_pred - flow_3d_target) ** 2, dim=0))
#                 gt_norm = torch.linalg.norm(flow_3d_target, axis=0)
#                 relative_err = epe3d_map / (gt_norm + 1e-4)

#                 acc3d_strict = torch.logical_or(epe3d_map < 0.05, relative_err < 0.05)
#                 acc3d_relax = torch.logical_or(epe3d_map < 0.1, relative_err < 0.1)
#                 outlier = torch.logical_or(epe3d_map > 0.3, relative_err > 0.1)

#                 metrics_3d['counts'] += epe3d_map.shape[0]
#                 metrics_3d['EPE3d'] += epe3d_map.sum().item()
#                 metrics_3d['AccS'] += torch.count_nonzero(acc3d_strict).item()
#                 metrics_3d['AccR'] += torch.count_nonzero(acc3d_relax).item()
#                 metrics_3d['Outlier'] += torch.count_nonzero(outlier).item()

#         logging.info('#### 3D Metrics ####')
#         logging.info('EPE: %.3f' % (metrics_3d['EPE3d'] / metrics_3d['counts']))
#         logging.info('AccS: %.2f%%' % (metrics_3d['AccS'] / metrics_3d['counts'] * 100.0))
#         logging.info('AccR: %.2f%%' % (metrics_3d['AccR'] / metrics_3d['counts'] * 100.0))
#         logging.info('Outlier: %.2f%%' % (metrics_3d['Outlier'] / metrics_3d['counts'] * 100.0))


# @hydra.main(config_path='conf', config_name='evaluator')
# def main(cfgs: DictConfig):
#     utils.init_logging()

#     # change working directory
#     shutil.rmtree(os.getcwd(), ignore_errors=True)
#     os.chdir(hydra.utils.get_original_cwd())

#     if torch.cuda.device_count() == 0:
#         device = torch.device('cpu')
#         logging.info('No CUDA device detected, using CPU for evaluation')
#     elif torch.cuda.device_count() == 1:
#         device = torch.device('cuda:0')
#         logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))
#         cudnn.benchmark = True
#     else:
#         raise RuntimeError('Evaluation script does not support multi-GPU systems.')

#     evaluator = Evaluator(device, cfgs)
#     evaluator.run()


# if __name__ == '__main__':
#     main()
