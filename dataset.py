import numpy as np
from irl_dcb.data import LHF_IRL, LHF_Human_Gaze
from irl_dcb.utils import compute_search_cdf, preprocess_fixations


def process_data(target_trajs,
                 DCB_HR_dir,
                 DCB_LR_dir,
                 target_annos,
                 hparams,
                 is_testing=False):
    target_init_fixs = {}
    for traj in target_trajs:
        key = traj['task'] + '_' + traj['name']
        target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                 traj['Y'][0] / hparams.Data.im_h)
    cat_names = list(np.unique([x['task'] for x in target_trajs]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    if is_testing:
        # testing fixation data
        test_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        assert len(test_target_trajs) > 0, 'no testing data found!'
        test_task_img_pair = np.unique(
            [traj['task'] + '_' + traj['name'] for traj in test_target_trajs])
        human_mean_cdf, _ = compute_search_cdf(test_target_trajs, target_annos,
                                               hparams.Data.max_traj_length)
        print('target fixation prob (test).:', human_mean_cdf)

        # load image data
        test_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                   test_task_img_pair, target_annos,
                                   hparams.Data, catIds)
        return {
            'catIds': catIds,
            'img_test': test_img_dataset,
            'human_mean_cdf': human_mean_cdf,
            'bbox_annos': target_annos,
            'gt_scanpaths': test_target_trajs
        }

    else:
        # training fixation data
        train_target_trajs = list(
            filter(lambda x: x['split'] == 'train', target_trajs))
        train_task_img_pair = np.unique(
            [traj['task'] + '_' + traj['name'] for traj in train_target_trajs])
        train_fix_labels = preprocess_fixations(
            train_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            truncate_num=hparams.Data.max_traj_length)

        # validation fixation data
        valid_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        valid_task_img_pair = np.unique(
            [traj['task'] + '_' + traj['name'] for traj in valid_target_trajs])
        human_mean_cdf, _ = compute_search_cdf(valid_target_trajs,
                                               target_annos,
                                               hparams.Data.max_traj_length)
        print('target fixation prob (valid).:', human_mean_cdf)
        valid_fix_labels = preprocess_fixations(
            valid_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            truncate_num=hparams.Data.max_traj_length)

        # load image data
        train_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                    train_task_img_pair, target_annos,
                                    hparams.Data, catIds)
        valid_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                    valid_task_img_pair, target_annos,
                                    hparams.Data, catIds)

        # load human gaze data
        train_HG_dataset = LHF_Human_Gaze(DCB_HR_dir, DCB_LR_dir,
                                          train_fix_labels, target_annos,
                                          hparams.Data, catIds)
        valid_HG_dataset = LHF_Human_Gaze(DCB_HR_dir,
                                          DCB_LR_dir,
                                          valid_fix_labels,
                                          target_annos,
                                          hparams.Data,
                                          catIds,
                                          blur_action=True)

        return {
            'catIds': catIds,
            'img_train': train_img_dataset,
            'img_valid': valid_img_dataset,
            'gaze_train': train_HG_dataset,
            'gaze_valid': valid_HG_dataset,
            'human_mean_cdf': human_mean_cdf,
            'bbox_annos': target_annos
        }
