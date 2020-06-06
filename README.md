# Scanpath_Prediction

PyTorch training/testing code and pretrained model for [Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning](https://arxiv.org/abs/2005.14310) (CVPR2020, oral)

We propose the first inverse reinforcement learning (IRL) model to learn the internal reward function and policy used by humans during visual search. The viewer's internal belief states were modeled as dynamic contextual belief maps of object locations. These maps were learned by IRL and then used to predict behavioral scanpaths for multiple target categories. To train and evaluate our IRL model we created COCO-Search18, which is now the largest dataset of high-quality search fixations in existence. COCO-Search18 has 10 participants searching for each of 18 target-object categories in 6202 images, making about 300,000 goal-directed fixations. When trained and evaluated on COCO-Search18, the IRL model outperformed baseline models in predicting search fixation scanpaths, both in terms of similarity to human search behavior and search efficiency.

If you are using this work, please cite:
```bibtex
@InProceedings{Yang_2020_CVPR_predicting,
author = {Yang, Zhibo and Huang, Lihan and Chen, Yupei and Wei, Zijun and Ahn, Seoyoung and Samaras, Dimitris and Zelinsky, Gregory and and Hoai, Minh},
title = {Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Scripts
- Train a model with
    ```
    python train.py <hparams> <dataset_root> [--cuda=<id>]
    ```
- Evaluate a trained model with
    ```
    python test.py <hparams> <checkpoint_dir> <dataset_root> [--cuda=<id>]
    ```
    
## Data Preparation
The dataset consists of two parts: image stimuli and fixations. For computational efficiency, we pre-compute the low- and high-resolution belief maps using the pretrained Panoptic FPN (with ResNet50 backbone) from [Detectron2](https://github.com/facebookresearch/detectron2).
For each image, we extract 134 beliefs maps for both low- and high-resolution and resize them to 20x32. Hence, for each image, we have two 134x20x32 tensors. Please refer to the [paper](https://arxiv.org/pdf/2005.14310.pdf) for more details.
Fixations come in the form of invidual scanpaths which mainly consists of a list of (x, y) locations in the image coordinate (see below for an example). Note that in the raw fixations there might be fixations out of the image boundaries, we remove them from the scanpaths.

The typical `<dataset_root>` should be structured as follows
```
<dataset_root>
    -- coco_search_annos_512x320.npy                    # bounding box annotation for each image (available at COCO)
    -- processed_human_scanpaths_TP_trainval.npy        # trainval split of human scanpaths (ground-truth)
    -- ./DCBs
        -- ./HR                                         # high-resolution belief maps of each input image (pre-computed)
        -- ./LR                                         # low-resolution belief maps of each input image (pre-computed)
```
The `processed_human_scanpaths_TP_trainval.npy` is a list of human scanpaths each of which is a `dict` object formated as follows
```
{
     'name': '000000400966.jpg',             # image name
     'subject': 2,                          # subject id
     'task': 'microwave',                   # target name
     'condition': 'present',                # target-present or target-absent
     'bbox': [67, 114, 78, 42],             # bounding box of the target object in the image
     'X': array([245.54666667, ...]),       # x-axis of each fixation
     'Y': array([128.03047619, ...]),       # y-axis of each fixation
     'T': array([190,  63, 180, 543]),      # duration of each fixation
     'length': 4,                           # length of the scanpath (i.e., number of fixations)
     'fixOnTarget': True,                   # if the scanpath lands on the target object
     'correct': 1,                          # 1 if the subject correctly located the target; 0 otherwise
     'split': 'train'                       # split of the image {'train', 'valid', 'test'}
 }
```
A sample `<dataset_root>` folder can be found at this [link](https://drive.google.com/open?id=1spD2_Eya5S5zOBO3NKILlAjMEC3_gKWc).

## COCO-Search18 Dataset
![coco-search18](./coco_search18_logo.png)

**COCO-Search18** dataset will be made available at https://saliency.tuebingen.ai/datasets/COCO-Search18/.

We are working on the release of the dataset, stay tuned!
