# Scanpath_Prediction
This is the source code (based on PyTorch and Python3) of the paper: [Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning](http://www3.cs.stonybrook.edu/~zhibyang/papers/scanpath-Pred_CVPR20.pdf) (CVPR2020, oral)

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
    
## COCO-Search18


**COCO-Search18** dataset will be made available at https://saliency.tuebingen.ai/datasets/COCO-Search18/.

We are working on the release of the dataset, stay tuned!
