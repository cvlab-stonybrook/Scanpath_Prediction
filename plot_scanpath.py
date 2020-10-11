import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import argparse
from os.path import isfile


def convert_coordinate(X, Y, im_w, im_h):
    """
    convert from display coordinate to pixel coordinate

    X - x coordinate of the fixations
    Y - y coordinate of the fixations
    im_w - image width
    im_h - image height
    """
    display_w, display_h = 1680, 1050
    target_ratio = display_w / float(display_h)
    ratio = im_w / float(im_h)

    delta_w, delta_h = 0, 0
    if ratio > target_ratio:
        new_w = display_w
        new_h = int(new_w / ratio)
        delta_h = display_h - new_h
    else:
        new_h = display_h
        new_w = int(new_h * ratio)
        delta_w = display_w - new_w
    dif_ux = delta_w // 2
    dif_uy = delta_h // 2
    scale = im_w / float(new_w)
    X = (X - dif_ux) * scale
    Y = (Y - dif_uy) * scale
    return X, Y


def plot_scanpath(img, xs, ys, ts, bbox=None, title=None):
    fig, ax = plt.subplots()
    ax.imshow(img)
    cir_rad_min, cir_rad_max = 30, 60
    min_T, max_T = np.min(ts), np.max(ts)
    rad_per_T = (cir_rad_max - cir_rad_min) / float(max_T - min_T)

    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                      ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)

    for i in range(len(xs)):
        cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            edgecolor='red',
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    if bbox is not None:
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
            alpha=0.5, edgecolor='yellow', facecolor='none', linewidth=2)
        ax.add_patch(rect)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_trial', choices=[0, 1],
                        default=1, type=int, help='randomly drawn from data (default=1)')
    parser.add_argument('--trial_id', type=int, default=1,
                        help='trial id (default=1)')
    parser.add_argument('--subj_id', type=int, default=2,
                        help='subject id (default=2)')
    parser.add_argument('--task', 
                        choices=['bot', 'chair', 'cup', 'fork', 'bowl', 'mouse',
                        'mic', 'lap', 'key', 'sink', 'toi', 'clock', 'tv', 'stop'], 
                        default='bot',
                        help='searching target')
    parser.add_argument('--condition', choices=[-1, 0, 1],
                        default=0, type=int, 
                        help='target present (1) or absent (-1) or randomly select (0) (default=0)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # load fixations data
    raw_data = pd.read_table('./fixation_report_s1-s10_full.xls')
    record = 's' + str(args.subj_id) + '_' + args.task
    data = raw_data[raw_data['RECORDING_SESSION_LABEL'] == record]
    if args.condition == 1:
        data = data[data['condition'] == 'present']
    elif args.condition == -1:
        data = data[data['condition'] == 'absent']

    if args.random_trial == 1:
        trial_ids = data['TRIAL_INDEX'].values
        tid = np.random.choice(trial_ids)
        data = data[data['TRIAL_INDEX'] == tid]
    else:
        data = data[data['TRIAL_INDEX'] == args.trial_id]
    if len(data) == 0:
        print("Error: no data found!")
        exit(-1)
    img_name = data['img_name'].values[0]
    cat_name = data['category'].values[0]
    if data['condition'].values[0] == 'absent':
        img_path = './images/TA/{}/{}'.format(cat_name, img_name)
        bbox = None
        print("This is target-absent trial")
    else:
        bboxes = np.load('./annos_1680x1050.npy').item()
        bbox = bboxes[cat_name + '_' + img_name]
        img_path = './images/TP/{}/{}'.format(cat_name, img_name)
        print("This is target-present trial")

    if not isfile(img_path):
        print("image not found at {}".format(img_path))
        exit(-1)

    # load image
    print(img_path)
    img = mpimg.imread(img_path)
    im_h, im_w = img.shape[0], img.shape[1]

    # convert fixations from display coordinate to pixel coordinate
    X, Y = data['CURRENT_FIX_X'].values.astype(
        np.float), data['CURRENT_FIX_Y'].values.astype(np.float)
    # X, Y = convert_coordinate(X, Y, im_w, im_h)
    T = data['CURRENT_FIX_DURATION'].values.astype(np.int32)

    title = "target-{}, target={}, correct={}".format(
        data['condition'].values[0], cat_name, data['Correct'].values[0])

    # plot_scanpath
    plot_scanpath(img, X, Y, T, bbox, title)
