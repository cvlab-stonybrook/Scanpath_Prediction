import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageFilter
import detectron2
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.engine import DefaultPredictor

device = torch.device("cuda")


def pred2feat(seg, info):
    seg = seg.cpu()
    feat = torch.zeros([80 + 54, 320, 512])
    for pred in info:
        mask = (seg == pred['id']).float()
        if pred['isthing']:
            feat[pred['category_id'], :, :] = mask * pred['score']
        else:
            feat[pred['category_id'] + 80, :, :] = mask
    return F.interpolate(feat.unsqueeze(0), size=[20, 32]).squeeze(0)


def get_DCBs(img_path, predictor, radius=1):
    high = Image.open(img_path).convert('RGB').resize((512, 320))
    low = high.filter(ImageFilter.GaussianBlur(radius=radius))
    high_panoptic_seg, high_segments_info = predictor(
        np.array(high))["panoptic_seg"]
    low_panoptic_seg, low_segments_info = predictor(
        np.array(low))["panoptic_seg"]
    high_feat = pred2feat(high_panoptic_seg, high_segments_info)
    low_feat = pred2feat(low_panoptic_seg, low_segments_info)
    return high_feat, low_feat


if __name__ == '__main__':

    # Load pretrained panoptic_fpn
    cfg = get_cfg()
    cfg.merge_from_file(
        '/home/zhibyang/github/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
    )
    model = build_backbone(cfg).to(device)
    model.eval()

    cfg.MODEL.WEIGHTS = 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'
    model_coco = build_backbone(cfg).to(device)
    model_coco.eval()
    predictor = DefaultPredictor(cfg)

    # Compute DCB
    img_path = '/home/zhibyang/projects/datasets/coco_search/images/320x512/TP/bottle/000000573206.jpg'
    high_feat, low_feat = get_DCBs(img_path, predictor)
    print(high_feat.shape, low_feat.shape)
