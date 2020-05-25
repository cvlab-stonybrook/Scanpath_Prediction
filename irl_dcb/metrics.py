import numpy as np
from .multimatch import docomparison
from . import utils


def multimatch(s1, s2, im_size):
    s1x = s1['X']
    s1y = s1['Y']
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
    s2x = s2['X']
    s2y = s2['Y']
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]


def compute_mm(human_trajs, model_trajs, im_w, im_h, tasks=None):
    """
    compute scanpath similarity using multimatch
    """
    all_mm_scores = []
    for traj in model_trajs:
        img_name = traj['name']
        task = traj['task']
        gt_trajs = list(
            filter(lambda x: x['name'] == img_name and x['task'] == task,
                   human_trajs))
        all_mm_scores.append((task,
                              np.mean([
                                  multimatch(traj, gt_traj, (im_w, im_h))[:4]
                                  for gt_traj in gt_trajs
                              ],
                                      axis=0)))

    if tasks is not None:
        mm_tasks = {}
        for task in tasks:
            mm = np.array([x[1] for x in all_mm_scores if x[0] == task])
            mm_tasks[task] = np.mean(mm, axis=0)
        return mm_tasks
    else:
        return np.mean([x[1] for x in all_mm_scores], axis=0)


def scanpath2clusters(meanshift, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    for i in range(len(xs)):
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)
    return string


def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))


# compute sequence score
def compute_SS(preds, clusters, truncate, reduce='mean'):
    results = []
    for scanpath in preds:
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for gt in strings:
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                gt = gt[:truncate] if len(gt) > truncate else gt
                score = nw_matching(pred, gt)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results


def get_seq_score(preds, clusters, max_step, tasks=None):
    results = compute_SS(preds, clusters, truncate=max_step)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))


def scanpath_ratio(traj, bbox):
    X1, Y1 = traj['X'][:-1], traj['Y'][:-1]
    X2, Y2 = traj['X'][1:], traj['Y'][1:]
    traj_dist = np.sum(np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2))
    cx, cy = traj['X'][0], traj['Y'][0]
    tx, ty = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
    target_dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
    if traj_dist == 0:
        print("error traj", traj)
    return min(target_dist / traj_dist, 1.0)


def compute_avgSPRatio(trajs, target_annos, max_step, tasks=None):

    all_sp_ratios = []
    for traj in trajs:
        key = traj['task'] + '_' + traj['name']
        bbox = target_annos[key]
        num_step = utils.get_num_step2target(traj['X'], traj['Y'], bbox)
        if num_step > max_step + 1:  # skip failed scanpaths
            continue
        sp = {'X': traj['X'][:num_step], 'Y': traj['Y'][:num_step]}
        if len(sp['X']) == 1:  # skip single-step scanpaths
            continue
        all_sp_ratios.append((traj['task'], scanpath_ratio(sp, bbox)))

    if tasks is not None:
        avg_sp_ratios = {}
        for task in tasks:
            sp_ratios = [x[1] for x in all_sp_ratios if x[0] == task]
            avg_sp_ratios[task] = np.mean(sp_ratios)
        return avg_sp_ratios
    else:
        return np.mean([x[1] for x in all_sp_ratios])


def compute_cdf_auc(cdf):
    if isinstance(cdf, dict):
        auc = {}
        for k, v in cdf.items():
            auc[k] = v[0] + v[-1] + np.sum(v[1:-1])
        return auc
    else:
        return cdf[0] + cdf[-1] + np.sum(cdf[1:-1])


def compute_prob_mismatch(cdf, human_mean_cdf):
    if isinstance(cdf, dict):
        return dict(
            zip(
                cdf.keys(),
                np.sum(np.abs(np.array(list(cdf.values())) - human_mean_cdf),
                       axis=1)))
    else:
        return np.sum(np.abs(cdf - human_mean_cdf))
