# cv_models_compression

## Model

[facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)

## Data

[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

## Evaluation

GPU: NVIDIA GeForce RTX 3050 4096 MiB

ONNXDQ_CPU: only 100 images

| Compression                   | Map    | Map_50 | Map_75 | FPS | Size, MB |
|:------------------------------|:-------|:-------|:-------|:----|:---------|
| None                          | 0.5615 | 0.7284 | 0.6085 | 11  | 158.404  |
| Quantization: PTDQ            | -      | -      | -      | -   | 91.633   |
| Quantization: ONNXDQ_CPU      | 0.5989 | 0.7582 | 0.6436 |0.18 | 40.67    |
| Prunning: Random Unstructured | 0.3023 | 0.4269 | 0.3252 | 11  | 158.404  |
| Prunning: L1 Unstructured     | 0.5607 | 0.7275 | 0.6069 | 11  | 158.404  |
| Prunning: Ln Unstructured     | 0.1291 | 0.2739 | 0.1131 | 11  | 158.404  |

Distillation with ResNet50 and Resnet18 as backbones and FasterRCNN as detector:

| Model    | Map    | Map_50 | Map_75 | FPS  | Size, MB | Params, amount |
|:---------|:-------|:-------|:-------|:-----|:---------|:---------------|
| ResNet50 | 0.2845 | 0.4204 | 0.3209 | 9.9  | 97.29    | 25,503,912     |
| ResNet18 | 0.0793 | 0.1658 | 0.0659 | 9.61 | 74.067   | 19,416,256     |

Weight sharing or clusterization by KMeans

| Compression                   | Map    | Map_50 | Map_75 | FPS | Size, MB |
|:------------------------------|:-------|:-------|:-------|:----|:---------|
| None                          | 0.5615 | 0.7284 | 0.6085 | 11  | 158.404  |
| Clustered model               | 0.4856 | 0.6441 | 0.5257 | 11  | 158.99   |
