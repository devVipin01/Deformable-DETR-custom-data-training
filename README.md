# Deformable DETR Object Detection (COCO)

This repository contains a **PyTorch Lightning** implementation of **Deformable DETR** for object detection, trained and evaluated on the **COCO dataset**.  
It uses **Hugging Face Transformers**, **Albumentations** for data augmentation, and **Supervision** for visualization.

---

## 🚀 Features

- Deformable DETR (`facebook/deformable-detr-detic`)
- COCO-style dataset support
- Advanced data augmentations with bounding boxes
- PyTorch Lightning training pipeline
- GPU acceleration support
- Inference + visualization of predictions vs ground truth

---

## 📁 Project Structure

```text
.
├── data/
│   └── coco/
│       ├── train2017/
│       ├── val2017/
│       └── instances_clean.json
├── train.py
├── requirements.txt
└── README.md
