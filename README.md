# cv_models_compression

## Model

[facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50)

## Data

[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

## Evaluation

GPU: NVIDIA GeForce RTX 3050 4096 MiB

| Compression                   | Map    | Map_50 | Map_75 | FPS | Size, MB |
|:------------------------------|:-------|:-------|:-------|:----|:---------|
| None                          | 0.5615 | 0.7284 | 0.6085 | 11  | 158.404  |
| Quantization: PTDQ            | -      | -      | -      | -   | 91.633   |
| Prunning: Random Unstructured | 0.3023 | 0.4269 | 0.3252 | 11  | 158.404  |
| Prunning: L1 Unstructured     | 0.5607 | 0.7275 | 0.6069 | 11  | 158.404  |
| Prunning: Ln Unstructured     | 0.1291 | 0.2739 | 0.1131 | 11  | 158.404  |

