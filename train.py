"""Train script.
Usage:
  train.py <hparams> <dataset_root> [--cuda=<id>]
  train.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>  Speed in knots [default: 0].
"""

import torch
import numpy as np
from docopt import docopt
from os.path import join
from irl_dcb.config import JsonConfig
from dataset import process_data
from irl_dcb.builder import build
from irl_dcb.trainer import Trainer
torch.manual_seed(42619)
np.random.seed(42619)

if __name__ == '__main__':
    args = docopt(__doc__)
    device = torch.device('cuda:{}'.format(args['--cuda']))
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
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
                         'processed_human_scanpaths_TP.npy')
    human_scanpaths = np.load(fixation_path,
                              allow_pickle=True,
                              encoding='latin1')

    # exclude incorrect scanpaths
    if hparams.Train.exclude_wrong_trials:
        human_scanpaths = list(filter(lambda x: x['correct'] == 1,
                                      human_scanpaths))

    # process fixation data
    dataset = process_data(human_scanpaths, DCB_dir_HR, DCB_dir_LR, bbox_annos,
                           hparams)

    built = build(hparams, True, device, dataset['catIds'])
    trainer = Trainer(**built, dataset=dataset, device=device, hparams=hparams)
    trainer.train()
