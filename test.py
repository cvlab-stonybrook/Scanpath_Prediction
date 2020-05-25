"""Train script.
Usage:
  test.py <hparams> <checkpoint_dir> <dataset_root> [--cuda=<id>]
  test.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>  Speed in knots [default: 0].
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from docopt import docopt
from os.path import join
from dataset import process_data
from irl_dcb.config import JsonConfig
from torch.utils.data import DataLoader
from irl_dcb.models import LHF_Policy_Cond_Small
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb import metrics
from irl_dcb import utils
torch.manual_seed(42619)
np.random.seed(42619)


def gen_scanpaths(generator,
                  env_test,
                  test_img_loader,
                  patch_num,
                  max_traj_len,
                  num_sample=10):
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader,
                        desc='trial ({}/{})'.format(i_sample + 1, num_sample))
        for i_batch, batch in enumerate(progress):
            env_test.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            with torch.no_grad():
                env_test.reset()
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            patch_num,
                                            max_traj_len,
                                            is_eval=True,
                                            sample_action=True)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])

    scanpaths = utils.actions2scanpaths(all_actions, patch_num)
    utils.cutFixOnTarget(scanpaths, bbox_annos)

    return scanpaths


if __name__ == '__main__':
    args = docopt(__doc__)
    device = torch.device('cuda:{}'.format(args['--cuda']))
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    checkpoint = args["<checkpoint_dir>"]
    hparams = JsonConfig(hparams)

    # dir of pre-computed beliefs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')
    data_name = '{}x{}'.format(hparams.Data.im_w, hparams.Data.im_h)

    # bounding box of the target object (for scanpath ratio evaluation)
    bbox_annos = np.load(join(dataset_root,
                              'coco_search_annos_{}.npy'.format(data_name)),
                         allow_pickle=True).item()

    # load ground-truth human scanpaths
    fixation_path = join(dataset_root,
                         'processed_human_scanpaths_TP_test.npy')
    human_scanpaths = np.load(fixation_path,
                              allow_pickle=True,
                              encoding='latin1')

    fix_clusters = np.load(join('./data', 'clusters.npy'),
                           allow_pickle=True).item()

    # process fixation data
    dataset = process_data(human_scanpaths,
                           DCB_dir_HR,
                           DCB_dir_LR,
                           bbox_annos,
                           hparams,
                           is_testing=True)
    img_loader = DataLoader(dataset['img_test'],
                            batch_size=64,
                            shuffle=False,
                            num_workers=16)
    print('num of test images =', len(dataset['img_test']))

    # load trained model
    input_size = 134  # number of belief maps
    task_eye = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      input_size).to(device)

    generator.load_state_dict(
        torch.load(join(checkpoint, 'trained_generator.pt'),
                   map_location=device))

    generator.eval()

    # build environment
    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True)

    # generate scanpaths
    print('sample scanpaths (10 for each testing image)...')
    predictions = gen_scanpaths(generator,
                                env_test,
                                img_loader,
                                hparams.Data.patch_num,
                                hparams.Data.max_traj_length,
                                num_sample=10)

    print('evaulating model...')
    # evaluate predictions
    mean_cdf, _ = utils.compute_search_cdf(predictions, bbox_annos,
                                           hparams.Data.max_traj_length)

    # scanpath ratio
    sp_ratio = metrics.compute_avgSPRatio(predictions, bbox_annos,
                                          hparams.Data.max_traj_length)

    # probability mismatch
    prob_mismatch = metrics.compute_prob_mismatch(mean_cdf,
                                                  dataset['human_mean_cdf'])

    # TFP-AUC
    tfp_auc = metrics.compute_cdf_auc(mean_cdf)

    # sequence score
    seq_score = metrics.get_seq_score(predictions, fix_clusters,
                                      hparams.Data.max_traj_length)

    # multimatch
    mm_score = metrics.compute_mm(dataset['gt_scanpaths'], predictions,
                                  hparams.Data.im_w, hparams.Data.im_h)

    # print and save outputs
    print('results:')
    results = {
        'cdf': list(mean_cdf),
        'sp_ratios': sp_ratio,
        'probability_mismatch': prob_mismatch,
        'TFP-AUC': tfp_auc,
        'sequence_score': seq_score,
        'multimatch': list(mm_score)
    }

    results = JsonConfig(results)
    save_path = join(checkpoint, '../results/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    results.dump(save_path)
    print('results successfully saved to {}'.format(save_path))
