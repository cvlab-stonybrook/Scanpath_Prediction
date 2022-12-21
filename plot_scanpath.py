import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse
import json
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
    parser.add_argument('--fixation_path', type=str, help='the path of the fixation json file')
    parser.add_argument('--image_dir', type=str, help='the directory of the image stimuli')
    parser.add_argument('--random_trial', choices=[0, 1],
                        default=1, type=int, help='randomly drawn from data (default=1)')
    parser.add_argument('--trial_id', default=0, type=int, help='trial id (default=0)')
    parser.add_argument('--subj_id', type=int, default=-1,
                        help='subject id (default=-1)')
    parser.add_argument('--task', 
                        choices=['bottle', 'chair', 'cup', 'fork', 'bowl', 'mouse',
                        'microwave', 'laptop', 'key', 'sink', 'toilet', 'clock', 'tv',
                        'stop sign', 'car', 'oven', 'knife'],
                        default='bottle',
                        help='searching target')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # load fixations data
    with open(args.fixation_path, 'r') as f:
        scanpaths = json.load(f)
    scanpaths = list(filter(lambda x: x['task'] == args.task, scanpaths))
    if args.subj_id > 0:
        scanpaths = list(filter(lambda x: x['subject'] == args.subj_id, scanpaths))
    
    if args.random_trial == 1:
        id = np.random.randint(len(scanpaths))
    else:
        id = args.trial_id
    scanpath = scanpaths[id]
    img_name = scanpath['name']
    cat_name = scanpath['task']
    bbox = scanpath['bbox']
    img_path = './{}/{}/{}'.format(args.image_dir, cat_name, img_name)
    print("This is target-present trial")

    if not isfile(img_path):
        print("image not found at {}".format(img_path))
        exit(-1)

    # load image
    print(img_path)
    img = mpimg.imread(img_path)
    im_h, im_w = img.shape[0], img.shape[1]

    # convert fixations from display coordinate to pixel coordinate
    X, Y, T = scanpath['X'], scanpath['Y'], scanpath['T']
    # X, Y = convert_coordinate(X, Y, im_w, im_h)

    title = "target={}, correct={}".format(cat_name, scanpath['correct'])

    # plot_scanpath
    plot_scanpath(img, X, Y, T, bbox, title)
