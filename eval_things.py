import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import utils
import hydra
import time
import shutil
import logging
import torch
import torch.optim
import torch.backends.cudnn as cudnn
import numpy as np
from thop import profile
from tqdm import tqdm
from omegaconf import DictConfig
from factory import model_factory, dataset_factory
from utils import copy_to_device, size_of_batch, save_flow_png, get_max_memory

# python eval_things.py testset=flyingthings3d_subset model=camliraft ckpt.path=checkpoints/camliraft_things150e.pt
# python eval_things.py testset=flyingthings3d_subset model=camliga ckpt.path=checkpoints/camliga_things150e.pt
# python eval_things.py testset=flyingthings3d_subset model=camliga ckpt.path=outputs/camliga/2024-01-03/22-27-54/ckpts/epoch-080.pt
# python eval_things.py testset=flyingthings3d_subset model=gaflow ckpt.path=checkpoints/GAFlow-CT2K.pth
# python eval_things.py testset=flyingthings3d_subset model=camliga ckpt.path=checkpoints/camligac_things145e_kitti790e.pt
# python eval_things.py testset=flyingthings3d_subset model=camliga ckpt.path=checkpoints/camligac_things145e.pt
# python eval_things.py testset=flyingthings3d_subset model=camliraft_l ckpt.path=outputs/camliraft_l/2024-05-09/16-37-26/ckpts/best.pt
# python eval_things.py testset=flyingthings3d_subset model=gmsf ckpt.path=/mnt/save_home_all/home/wxy/code/camliga/outputs/gmsf/2024-06-24/10-56-18/ckpts/best.pt

class Evaluator:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
        self.test_dataset = dataset_factory(self.cfgs.testset)
        self.test_loader = utils.FastDataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfgs.testset.batch_size,
            num_workers=self.cfgs.testset.n_workers
        )

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

    @torch.no_grad()
    def run(self):
        logging.info('Running evaluation...')
        self.model.eval()

        metrics_2d = {'counts': 0, 'EPE2d': 0.0, '1px': 0.0, 'Fl': 0.0}
        metrics_3d = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}
        metrics_3d_noc = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}
        epe2d_one_list = []

        for inputs in tqdm(self.test_loader):
            
            inputs = copy_to_device(inputs, self.device)
            # print(inputs.keys())  # 打印字典中的所有键
            # flops, params = profile(self.model, inputs=(inputs,))
            
            # start_time = time.time()
            

            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.forward(inputs)
            
            # timing = time.time() - start_time
            # start_time = time.time()
            # mem = get_max_memory(self.device, torch.cuda.device_count())

            for batch_id in range(size_of_batch(inputs)):
                if 'flow_2d' in outputs:
                    flow_2d_pred = outputs['flow_2d'][batch_id]
                    flow_2d_target = inputs['flow_2d'][batch_id]

                    if flow_2d_target.shape[0] > 2:
                        flow_2d_mask = flow_2d_target[2] > 0
                        flow_2d_target = flow_2d_target[:2]
                    else:
                        flow_2d_mask = torch.ones(flow_2d_target.shape[1:], dtype=torch.int64, device=self.device)

                    epe2d_map = torch.sqrt(torch.sum((flow_2d_pred - flow_2d_target) ** 2, dim=0))
                    flow_2d_mask = torch.logical_and(flow_2d_mask, torch.logical_not(torch.isnan(epe2d_map)))
                    flow_2d_target_mag = torch.linalg.norm(flow_2d_target, dim=0)
                    fl_err_map = torch.logical_and(epe2d_map > 3.0, epe2d_map / flow_2d_target_mag > 0.05)

                    metrics_2d['counts'] += epe2d_map[flow_2d_mask].shape[0]
                    '''
                    testepe_id = inputs['index'][batch_id]
                    epe2d_all = epe2d_map[flow_2d_mask].sum().item()
                    epe2d_one = epe2d_all / epe2d_map[flow_2d_mask].shape[0]
                    if True:
                        print("test_id", testepe_id)
                        print('EPE: %.3f' % (epe2d_one))
                        epe2d_one_list.append(epe2d_one)
                    '''
                    metrics_2d['EPE2d'] += epe2d_map[flow_2d_mask].sum().item()
                    metrics_2d['1px'] += torch.count_nonzero(epe2d_map[flow_2d_mask] < 1.0).item()
                    metrics_2d['Fl'] += fl_err_map[flow_2d_mask].float().sum().item()
                    # logging.info('EPE2d: %.1f, time: %.2fs, mem: %dM, flops:%.2f' % (epe2d_map[flow_2d_mask].sum().item(), timing, mem, flops))


                    if self.cfgs.save_results:
                        test_id = inputs['index'][batch_id]
                        os.makedirs('prediction/things/flow_2d', exist_ok=True)
                        flow_2d_pred = flow_2d_pred.clamp(-500, 500).permute(1, 2, 0).cpu().numpy()
                        save_flow_png('prediction/things/flow_2d/%07d.png' % test_id, flow_2d_pred)

                if 'flow_3d' in outputs:
                    flow_3d_pred = outputs['flow_3d'][batch_id]
                    flow_3d_target = inputs['flow_3d'][batch_id]

                    if flow_3d_target.shape[0] > 3:
                        flow_3d_mask = flow_3d_target[3] > 0
                        flow_3d_target = flow_3d_target[:3]
                    else:
                        flow_3d_mask = torch.ones(flow_3d_target.shape[1], dtype=torch.int64, device=self.device)

                    epe3d_map = torch.sqrt(torch.sum((flow_3d_pred - flow_3d_target) ** 2, dim=0))
                    flow_3d_mask = torch.logical_and(flow_3d_mask, torch.logical_not(torch.isnan(epe3d_map)))

                    metrics_3d['counts'] += epe3d_map[flow_3d_mask].shape[0]
                    metrics_3d['EPE3d'] += epe3d_map[flow_3d_mask].sum().item()
                    # metrics_3d['5cm'] += torch.count_nonzero(epe3d_map[flow_3d_mask] < 0.05).item()
                    # metrics_3d['10cm'] += torch.count_nonzero(epe3d_map[flow_3d_mask] < 0.1).item()
                    # metrics_3d['outlier'] += torch.count_nonzero(epe3d_map[flow_3d_mask] > 0.3).item()
                    # another way to calculate 5cm and 10cm
                    metrics_3d['5cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.05), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] < 0.05))).item()
                    metrics_3d['10cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.1), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] < 0.1))).item()
                    metrics_3d['outlier'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] > 0.3), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] > 0.1))).item()

                    # evaluate on non-occluded points
                    if 'occ_mask_3d' in inputs:
                        occ_mask_3d = inputs['occ_mask_3d'][batch_id]
                        flow_3d_mask = torch.logical_and(occ_mask_3d == 0, flow_3d_mask)
                        # epe3d_map_noc = epe3d_map[torch.logical_and(occ_mask_3d == 0, flow_3d_mask)]
                        epe3d_map_noc = epe3d_map[flow_3d_mask]
                        # flow_3d_target_noc = flow_3d_target[:, flow_3d_mask]  # 应用掩码
                        metrics_3d_noc['counts'] += epe3d_map_noc.shape[0]
                        metrics_3d_noc['EPE3d'] += epe3d_map_noc.sum().item()
                        metrics_3d_noc['5cm'] += torch.count_nonzero(epe3d_map_noc < 0.05).item()
                        metrics_3d_noc['10cm'] += torch.count_nonzero(epe3d_map_noc < 0.1).item()
                        metrics_3d_noc['outlier'] += torch.count_nonzero(epe3d_map_noc > 0.3).item()
                        # another way to calculate 5cm and 10cm
                        # metrics_3d_noc['5cm'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc < 0.05), (epe3d_map_noc/torch.sqrt(torch.sum((flow_3d_target_noc) ** 2, dim=0))[flow_3d_mask] < 0.05))).item()
                        # metrics_3d_noc['10cm'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc < 0.1), (epe3d_map_noc/torch.sqrt(torch.sum((flow_3d_target_noc) ** 2, dim=0))[flow_3d_mask] < 0.1))).item()
                        # metrics_3d_noc['outlier'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc > 0.3), (epe3d_map_noc/torch.sqrt(torch.sum((flow_3d_target_noc) ** 2, dim=0))[flow_3d_mask] > 0.1))).item()

                    if self.cfgs.save_results:
                        test_id = inputs['index'][batch_id]
                        os.makedirs('prediction/things/flow_3d', exist_ok=True)
                        flow_3d_pred = flow_3d_pred.transpose(0, 1).cpu().numpy()
                        np.save('prediction/things/flow_3d/%07d.npy' % test_id, flow_3d_pred)

        if metrics_2d['counts'] > 0:
            # np.savetxt('prediction/things/epe2d.txt', epe2d_one_list, fmt='%.3f', delimiter='\n')
            logging.info('#### 2D Metrics ####')
            logging.info('EPE: %.4f' % (metrics_2d['EPE2d'] / metrics_2d['counts']))
            logging.info('1px: %.2f%%' % (metrics_2d['1px'] / metrics_2d['counts'] * 100.0))
            logging.info('Fl:  %.2f%%' % (metrics_2d['Fl'] / metrics_2d['counts'] * 100.0))

        if metrics_3d['counts'] > 0:
            logging.info('#### 3D Metrics ####')
            logging.info('EPE: %.4f' % (metrics_3d['EPE3d'] / metrics_3d['counts']))
            logging.info('5cm: %.2f%%' % (metrics_3d['5cm'] / metrics_3d['counts'] * 100.0))
            logging.info('10cm: %.2f%%' % (metrics_3d['10cm'] / metrics_3d['counts'] * 100.0))
            logging.info('outlier: %.4f%%' % (metrics_3d['outlier'] / metrics_3d['counts'] * 100.0))

        if metrics_3d_noc['counts'] > 0:
            logging.info('#### 3D Metrics (Non-occluded) ####')
            logging.info('EPE: %.4f' % (metrics_3d_noc['EPE3d'] / metrics_3d_noc['counts']))
            logging.info('5cm: %.2f%%' % (metrics_3d_noc['5cm'] / metrics_3d_noc['counts'] * 100.0))
            logging.info('10cm: %.2f%%' % (metrics_3d_noc['10cm'] / metrics_3d_noc['counts'] * 100.0))
            logging.info('outlier: %.4f%%' % (metrics_3d_noc['outlier'] / metrics_3d_noc['counts'] * 100.0))


@hydra.main(config_path='conf', config_name='evaluator')
def main(cfgs: DictConfig):
    utils.init_logging()

    # change working directory
    shutil.rmtree(os.getcwd(), ignore_errors=True)
    os.chdir(hydra.utils.get_original_cwd())

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))
        cudnn.benchmark = True
    else:
        raise RuntimeError('Evaluation script does not support multi-GPU systems.')

    evaluator = Evaluator(device, cfgs)
    evaluator.run()


if __name__ == '__main__':
    main()
