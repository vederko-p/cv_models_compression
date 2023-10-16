import os
import time
from typing import List, Tuple
from collections import namedtuple
from bs4 import BeautifulSoup, element
from tqdm.notebook import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchmetrics.detection import MeanAveragePrecision
from transformers import DetrImageProcessor


# Parse images and annots utils

Box = namedtuple('Box', 'xmin ymin xmax ymax')
VOCObject = namedtuple('VOCObject', 'cls xyxy')


def read_set(filepath: str) -> List[List[str]]:
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [line[:-1] for line in lines]


def read_xml(filepath: str) -> BeautifulSoup:
    with open(filepath, 'r') as xml_file:
        data = xml_file.read()
    return BeautifulSoup(data, 'xml')


def read_image_rgb(filepath: str) -> np.array:
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def parse_box_xyxy(annot_obj: element.Tag) -> Box:
    return Box(
        int(annot_obj.xmin.string),
        int(annot_obj.ymin.string),
        int(annot_obj.xmax.string),
        int(annot_obj.ymax.string),
    )


def parse_objects(annot: BeautifulSoup) -> List[VOCObject]:
    return [
        VOCObject(obj.find('name').string, parse_box_xyxy(obj))
        for obj in annot.find_all('object')
    ]


# Parse model results utils

def parse_to_voc_objects(model_res: dict, model_id2lbl: dict) -> List[VOCObject]:
    """
    model_res: dict results from processor.post_process_object_detection
    """
    labels = [model_id2lbl[lbl_id] for lbl_id in model_res['labels'].cpu().tolist()]
    boxes = [Box(*box.to(int).tolist()) for box in model_res['boxes'].cpu().detach()]
    return [VOCObject(cls, box) for cls, box in zip(labels, boxes)]


def draw_voc_objects(image: np.array, objects: List[VOCObject]) -> None:
    color = (0, 255, 255)
    image_draw = image.copy()
    for obj in objects:
        tl = (obj.xyxy.xmin, obj.xyxy.ymin)
        br = (obj.xyxy.xmax, obj.xyxy.ymax)        
        cv2.rectangle(image_draw, tl, br, color, 1)
        cv2.putText(
            image_draw, obj.cls, tl,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA
        )
    plt.imshow(image_draw);


# Evaluate model utils

coco2voc = {
    'airplane': 'aeroplane',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'boat': 'boat',
    'bottle': 'bottle',
    'bus': 'bus',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'cow': 'cow',
    'dining table': 'diningtable',
    'dog': 'dog',
    'horse': 'horse',
    'motorcycle': 'motorbike',
    'person': 'person',
    'potted plant': 'pottedplant',
    'sheep': 'sheep',
    'couch': 'sofa',
    'train': 'train',
    'tv': 'tvmonitor',
}

voc2_id2label = {
    0: 'aeroplane',
    1: 'bicycle',
    2: 'bird',
    3: 'boat',
    4: 'bottle',
    5: 'bus',
    6: 'car',
    7: 'cat',
    8: 'chair',
    9: 'cow',
    10: 'diningtable',
    11: 'dog',
    12: 'horse',
    13: 'motorbike',
    14: 'person',
    15: 'pottedplant',
    16: 'sheep',
    17: 'sofa',
    18: 'train',
    19: 'tvmonitor',
}

voc2_label2id = {lbl: i  for i, lbl in voc2_id2label.items()}


def cocolbl2voc(coco_lbl: int, model_id2lbl: dict) -> int:
    return voc2_label2id[coco2voc[model_id2lbl[coco_lbl]]]


def results2pred(results: dict, model_id2lbl: dict) -> dict:
    voc_lbls_mask = torch.tensor([
        model_id2lbl[lbl] in coco2voc.keys()
        for lbl in results['labels'].tolist()
    ])
    if voc_lbls_mask.any():
        return {
            'boxes': results['boxes'][voc_lbls_mask].detach(),
            'scores': results['scores'][voc_lbls_mask].detach(),
            'labels': torch.tensor([
                cocolbl2voc(lbl, model_id2lbl)
                for lbl in results['labels'][voc_lbls_mask].tolist()
            ]),
        }
    else:
        return {
            'boxes': torch.tensor([]).to(float),
            'scores': torch.tensor([]).to(float),
            'labels': torch.tensor([]),
        }


def voc_objects2target(voc_objects: List[VOCObject]) -> dict:
    gt_boxes, gt_labels = zip(*[
        (list(voc_obj.xyxy), voc2_label2id[voc_obj.cls])
        for voc_obj in voc_objects
    ])
    return {
        'boxes': torch.tensor(gt_boxes).to(float),
        'labels': torch.tensor(gt_labels)
    }

def evaluate_over_voc(
    images_path: str,
    annots_path: str,
    val_images: List[str],
    model: torch.nn.Module,
    proc: DetrImageProcessor,
    device: str = None,
) -> Tuple[dict, float]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    map = MeanAveragePrecision(iou_type='bbox')
    eval_time_sec = []
    for img_name in tqdm(val_images, total=len(val_images)):
    
        img_p = os.path.join(images_path, f'{img_name}.jpg')
        annot_p = os.path.join(annots_path, f'{img_name}.xml')
        val_img = read_image_rgb(img_p)
        val_annot = parse_objects(read_xml(annot_p))
        
        inputs = proc(images=val_img, return_tensors="pt").to(device)
        st = time.time()
        outputs = model(**inputs)
        et = time.time()
        eval_time_sec.append(et - st)
        target_sizes = torch.tensor([val_img.shape[:-1]])
        results = proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        preds = [results2pred(results, model.config.id2label)]
        targets = [voc_objects2target(val_annot)]
        map.update(preds, targets)

    return map.compute(), round(sum(eval_time_sec) / len(eval_time_sec), 3)


def model_size(model):
    """
    Return model size in MB.
    """
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / 1024**2


# ONNX and OpenVino utils

class MyOutput:
    def __init__(self, logits, pred_boxes):
        self.logits = torch.from_numpy(logits)
        self.pred_boxes = torch.from_numpy(pred_boxes)


def evaluate_over_voc_onnx(
        images_path: str,
        annots_path: str,
        val_images: List[str],
        model: torch.nn.Module,
        proc: DetrImageProcessor,
        torch_model: torch.nn.Module,
        ort_session,
) -> Tuple[dict, float]:
    map = MeanAveragePrecision(iou_type='bbox')
    eval_time_sec = []
    for img_name in tqdm(val_images, total=len(val_images)):

        img_p = os.path.join(images_path, f'{img_name}.jpg')
        annot_p = os.path.join(annots_path, f'{img_name}.xml')
        val_img = read_image_rgb(img_p)
        val_annot = parse_objects(read_xml(annot_p))

        coef_h, coef_w = [800, 1137] / np.array(val_img.shape[:2])
        val_img = cv2.resize(val_img, (1137, 800),
                             interpolation=cv2.INTER_LINEAR)
        inputs = proc(images=val_img, return_tensors=None)

        st = time.time()
        outputs = model.run(None, {
            ort_session.get_inputs()[0].name: inputs.pixel_values})
        et = time.time()
        eval_time_sec.append(et - st)

        target_sizes = torch.tensor([val_img.shape[:-1]])
        outputs = MyOutput(outputs[0], outputs[1])
        results = \
        proc.post_process_object_detection(outputs, target_sizes=target_sizes,
                                           threshold=0.9)[0]

        preds = [results2pred(results, torch_model.config.id2label)]
        targets = [voc_objects2target(val_annot)]

        if len(preds[0]['boxes']) > 0:
            boxes = torch.round(torch.hstack([
                preds[0]['boxes'][:, [0]] / coef_w,
                preds[0]['boxes'][:, [1]] / coef_h,
                preds[0]['boxes'][:, [2]] / coef_w,
                preds[0]['boxes'][:, [3]] / coef_h,
            ]))
            preds[0]['boxes'] = boxes

        map.update(preds, targets)

    return map.compute(), round(sum(eval_time_sec) / len(eval_time_sec), 3)


def evaluate_over_voc_ov(
        images_path: str,
        annots_path: str,
        val_images: List[str],
        model: torch.nn.Module,
        proc: DetrImageProcessor,
        torch_model: torch.nn.Module
) -> Tuple[dict, float]:
    map = MeanAveragePrecision(iou_type='bbox')
    eval_time_sec = []
    for img_name in tqdm(val_images, total=len(val_images)):
        img_p = os.path.join(images_path, f'{img_name}.jpg')
        annot_p = os.path.join(annots_path, f'{img_name}.xml')
        val_img = read_image_rgb(img_p)
        val_annot = parse_objects(read_xml(annot_p))

        coef_h, coef_w = [800, 1137] / np.array(val_img.shape[:2])
        val_img = cv2.resize(val_img, (1137, 800),
                             interpolation=cv2.INTER_LINEAR)
        inputs = proc(images=val_img, return_tensors=None)

        st = time.time()
        outputs = model(inputs.pixel_values)
        et = time.time()
        eval_time_sec.append(et - st)

        target_sizes = torch.tensor([val_img.shape[:-1]])
        outputs = MyOutput(outputs[model.output(0)], outputs[model.output(1)])
        results = \
        proc.post_process_object_detection(outputs, target_sizes=target_sizes,
                                           threshold=0.9)[0]
        preds = [results2pred(results, torch_model.config.id2label)]
        targets = [voc_objects2target(val_annot)]

        if len(preds[0]['boxes']) > 0:
            boxes = torch.round(torch.hstack([
                preds[0]['boxes'][:, [0]] / coef_w,
                preds[0]['boxes'][:, [1]] / coef_h,
                preds[0]['boxes'][:, [2]] / coef_w,
                preds[0]['boxes'][:, [3]] / coef_h,
            ]))
            preds[0]['boxes'] = boxes

        map.update(preds, targets)

    return map.compute(), round(sum(eval_time_sec) / len(eval_time_sec), 3)
