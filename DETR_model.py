# import numpy as np
import os
import random
import numpy as np
from PIL import Image
import cv2 as cv
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import DeformableDetrImageProcessor, DeformableDetrForObjectDetection
import supervision as sv
import matplotlib.pyplot as plt
from torchinfo import summary

# =========================
# Dataset paths
# =========================
dataset = r"E:\Lenskart\Deformable_DETR\data\coco"
ANNOTATION_FILE_NAME = "instances_clean.json"

TRAIN_DIR = os.path.join(dataset, "train2017")
VAL_DIR = os.path.join(dataset, "val2017")
TEST_DIR = os.path.join(dataset, "val2017")  # using val as test for now

# =========================
# Augmentations
# =========================
def create_augmentations(height=800, width=1333):
    return A.Compose([
        A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.GaussianBlur(p=0.2),
        A.CLAHE(p=0.2),
        A.HueSaturationValue(p=0.2),
    ], bbox_params=A.BboxParams(
        format='coco',           # xywh in pixels
        label_fields=['category_ids'],
        min_visibility=0.0       # keep partially visible boxes
    ))

def augment_data(image, target, augmentations):
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    bboxes = []
    category_ids = []

    for ann in target:
        x, y, bw, bh = ann['bbox']
        # Clip bbox to image boundaries
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        bboxes.append([x, y, bw, bh])
        category_ids.append(ann['category_id'])

    # Skip images with no boxes
    if len(bboxes) == 0:
        return image, target

    augmented = augmentations(image=image_np, bboxes=bboxes, category_ids=category_ids)
    augmented_image = Image.fromarray(augmented['image'])

    for i, bbox in enumerate(augmented['bboxes']):
        target[i]['bbox'] = list(bbox)

    return augmented_image, target

# =========================
# Custom COCO Dataset
# =========================
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_dir_path: str, processor, train: bool=True):
        ann_file = os.path.join(img_dir_path, ANNOTATION_FILE_NAME)
        super().__init__(img_dir_path, ann_file)
        self.processor = processor
        self.train = train
        self.augmentations = create_augmentations() if train else None

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        if self.train and self.augmentations:
            img, target = augment_data(img, target, self.augmentations)

        target_dict = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target_dict, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        target = encoding["labels"][0]

        return pixel_values, target

# =========================
# Processor & Dataset
# =========================
CHECK_POINT = 'facebook/deformable-detr-detic'
processor = DeformableDetrImageProcessor.from_pretrained(
    CHECK_POINT,
    size={"shortest_edge": 800, "longest_edge": 1333}
)

train_dataset = CocoDetection(TRAIN_DIR, processor=processor)
val_dataset = CocoDetection(VAL_DIR, processor=processor, train=False)
test_dataset = CocoDetection(TEST_DIR, processor=processor, train=False)

# =========================
# DataLoader
# =========================
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

BATCH_SIZE = 16
NUM_WORKERS = 0
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# =========================
# Model Definition
# =========================
categories = test_dataset.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()
print("Class mapping:", id2label)

class DeformableDetr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DeformableDetrForObjectDetection.from_pretrained(
            CHECK_POINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k,v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n,p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n,p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone}
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

# =========================
# Initialize and summarize model
# =========================
model_params = {'lr':1e-4, 'lr_backbone':1e-5, 'weight_decay':1e-4}
model = DeformableDetr(**model_params)
summary(model, col_names=['trainable'])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# Trainer
# =========================
from pytorch_lightning import Trainer
trainer = Trainer(devices=1, accelerator='gpu', max_epochs=70, accumulate_grad_batches=4, log_every_n_steps=20)
trainer.fit(model)

# =========================
# Inference & Visualization
# =========================
image_ids = test_dataset.coco.getImgIds()
image_id = random.choice(image_ids)
image = test_dataset.coco.loadImgs(image_id)[0]
annotations = test_dataset.coco.imgToAnns[image_id]
image_path = os.path.join(test_dataset.root, image['file_name'])
image = cv.imread(image_path)

# Ground truth
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
labels = [f"{id2label[_]}" for _ in detections.class_id]
frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# Prediction
model.to(device)
with torch.inference_mode():
    inputs = processor(images=image, return_tensors='pt').to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.085, target_sizes=target_sizes)[0]
    detections = sv.Detections.from_transformers(transformers_results=results)
    labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
    frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# Show
plt.figure(figsize=(24, 16))
plt.imshow(cv.cvtColor(frame_ground_truth, cv.COLOR_BGR2RGB))
plt.title("Ground Truth")
plt.show()

plt.figure(figsize=(32, 16))
plt.imshow(cv.cvtColor(frame_detections, cv.COLOR_BGR2RGB))
plt.title("Prediction")
plt.show()


##############################################################################
# import pandas as pd
# import random
# import os
# from PIL import Image
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import transforms
# import pytorch_lightning as pl
# import scipy
# import cv2 as cv
# import supervision as sv
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from transformers import DeformableDetrImageProcessor
# from torch.utils.data import DataLoader
# import pytorch_lightning as pl
# from transformers import DeformableDetrForObjectDetection
# from torchinfo import summary
# import matplotlib.pyplot as plt

# ####data set config
# dataset = "E:\Lenskart\Deformable_DETR\data\coco"

# ANNOTATION_FILE_NAME = "instances_clean.json"
# TRAIN_DIR = os.path.join(dataset, "train2017")
# VAL_DIR = os.path.join(dataset, "val2017")
# TEST_DIR = os.path.join(dataset, "val2017")#"test"


# ####coustom coco dataset class
# def create_augmentations(height=800, width=1333):
#     return A.Compose([
#         A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.RandomGamma(p=0.2),
#         A.GaussianBlur(p=0.2),
#         A.CLAHE(p=0.2),
#         A.HueSaturationValue(p=0.2),
#     ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# def augment_data(image, target, augmentations):
#     # Convert PIL Image to numpy array
#     image_np = np.array(image)
    
#     # Extract bounding boxes and category IDs
#     bboxes = [ann['bbox'] for ann in target]
#     category_ids = [ann['category_id'] for ann in target]
    
#     # Apply augmentations
#     augmented = augmentations(image=image_np, bboxes=bboxes, category_ids=category_ids)
    
#     # Update image and target with augmented data
#     augmented_image = Image.fromarray(augmented['image'])
#     for i, bbox in enumerate(augmented['bboxes']):
#         target[i]['bbox'] = list(bbox)
    
#     return augmented_image, target

# ##
# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_dir_path: str, processor, train: bool=True):
#         ann_file = os.path.join(img_dir_path, ANNOTATION_FILE_NAME)
#         super(CocoDetection, self).__init__(img_dir_path, ann_file)
#         self.processor = processor
#         self.train = train
#         self.augmentations = create_augmentations() if train else None
        
#     def __getitem__(self, idx):
#         img, target = super(CocoDetection, self).__getitem__(idx)
#         image_id = self.ids[idx]
        
#         if self.train and self.augmentations:
#             img, target = augment_data(img, target, self.augmentations)
            
#         target = {'image_id': image_id, 'annotations': target}
#         encoding = self.processor(images=img, annotations=target, return_tensors="pt")
#         pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
#         target = encoding["labels"][0] # remove batch dimension
 
#         return pixel_values, target
    
# ###data preprocessing###

# CHECK_POINT = 'facebook/deformable-detr-detic'
# processor = DeformableDetrImageProcessor.from_pretrained(
#     CHECK_POINT,
#     size={"shortest_edge": 800, "longest_edge": 1333}
# )


# train_dataset = CocoDetection(TRAIN_DIR, processor=processor)
# val_dataset = CocoDetection(VAL_DIR, processor=processor, train=False)
# test_dataset = CocoDetection(TEST_DIR, processor=processor, train=False)

# ###data loader setup####
# # set up classes of target and create instance box annotator for drawing box in image
# categories = test_dataset.coco.cats
# id2label = {k: v['name'] for k,v in categories.items()}
# box_annotator = sv.BoxAnnotator()
# print(id2label)


# def collate_fn(batch):
#     pixel_values = [item[0] for item in batch]
#     encoding = processor.pad(pixel_values, return_tensors="pt")
#     labels = [item[1] for item in batch]
#     batch = {}
#     batch['pixel_values'] = encoding['pixel_values']
#     batch['pixel_mask'] = encoding['pixel_mask']
#     batch['labels'] = labels
#     return batch

# BATCH_SIZE = 2
# NUM_WORKERS = 0
# train_dataloader = DataLoader(train_dataset,
#                               collate_fn=collate_fn,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True,
#                               num_workers=NUM_WORKERS)

# val_dataloader = DataLoader(val_dataset,
#                             collate_fn=collate_fn,
#                             batch_size=BATCH_SIZE,
#                             num_workers=NUM_WORKERS)

# test_dataloader = DataLoader(test_dataset,
#                              collate_fn=collate_fn,
#                              batch_size=BATCH_SIZE,
#                              num_workers=NUM_WORKERS)
# batch = next(iter(train_dataloader)) # check if any error raise


# #####model defination###################

# class DeformableDetr(pl.LightningModule):
#     def __init__(self, lr, lr_backbone, weight_decay):
#         super().__init__()
#         self.model = DeformableDetrForObjectDetection.from_pretrained(CHECK_POINT,
#                                                          num_labels=len(id2label),
#                                                          ignore_mismatched_sizes=True)
#         self.lr = lr
#         self.lr_backbone = lr_backbone
#         self.weight_decay = weight_decay

#     def forward(self, pixel_values, pixel_mask):
#         outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

#         return outputs

#     def common_step(self, batch, batch_idx):
#         pixel_values = batch["pixel_values"]
#         pixel_mask = batch["pixel_mask"]
#         labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

#         outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

#         loss = outputs.loss
#         loss_dict = outputs.loss_dict

#         return loss, loss_dict

#     def training_step(self, batch, batch_idx):
#         loss, loss_dict = self.common_step(batch, batch_idx)
#         # logs metrics for each training_step,
#         # and the average across the epoch
#         self.log("training_loss", loss)
#         for k,v in loss_dict.items():
#             self.log("train_" + k, v.item())

#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, loss_dict = self.common_step(batch, batch_idx)
#         self.log("validation_loss", loss)
#         for k,v in loss_dict.items():
#             self.log("validation_" + k, v.item())

#         return loss

#     def configure_optimizers(self):
#         param_dicts = [
#               {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
#               {
#                   "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
#                   "lr": self.lr_backbone,
#               },
#         ]
#         optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
#                                   weight_decay=self.weight_decay)

#         return optimizer

#     def train_dataloader(self):
#         return train_dataloader

#     def val_dataloader(self):
#         return val_dataloader
    
    
# ################model initlization and layer freezing############

# model_params = {
#     'lr': 1e-4,
#     'lr_backbone': 1e-5,
#     'weight_decay': 1e-4
# }


# model = DeformableDetr(**model_params)
# summary(model, col_names=['trainable'])


# ########training########
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# from pytorch_lightning import Trainer

# trainer_params = {
#     'devices': 1,
#     'accelerator': 'gpu',
#     'max_epochs': 70,
#     'accumulate_grad_batches': 4,
#     'log_every_n_steps': 20
# }
# trainer = Trainer(**trainer_params)

# trainer.fit(model)

# ####model save 
# MODEL_PATH = '/model/'
# # model.model.save_pretrained(MODEL_PATH, safe_serialization=False)
# model.model.load_state_dict(torch.load('/model/pytorch_model.bin', weights_only=True))


# ####infer and visulization############
# image_ids = test_dataset.coco.getImgIds()
# image_id = random.choice(image_ids)
# print('Image #{}'.format(image_id))

# # load image and its annotations
# image = test_dataset.coco.loadImgs(image_id)[0]
# annotations = test_dataset.coco.imgToAnns[image_id]
# image_path = os.path.join(test_dataset.root, image['file_name'])
# image = cv.imread(image_path)

# # annotate ground truth
# detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
# labels = [f"{id2label[_]}" for _ in detections.class_id]
# frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections, 
#                                             labels=labels)


# # predict and annotate detections
# model.to(device)
# confindence_threshold = 0.085
# with torch.inference_mode():

#     # load image and predict
#     inputs = processor(images=image, return_tensors='pt').to(device)
#     outputs = model(**inputs)

#     # post-process
#     target_sizes = torch.tensor([image.shape[:2]]).to(device)
#     results = processor.post_process_object_detection(
#         outputs=outputs, 
#         threshold=confindence_threshold, 
#         target_sizes=target_sizes
#     )[0]


#     detections = sv.Detections.from_transformers(transformers_results=results)
#     labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
#     frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    
    
# #Show
# plt.figure(figsize=(24, 16))
# plt.imshow(cv.cvtColor(frame_ground_truth, cv.COLOR_BGR2RGB))
# plt.title("Ground Truth")
# plt.show()

# plt.figure(figsize=(32, 16))
# plt.imshow(cv.cvtColor(frame_detections, cv.COLOR_BGR2RGB))
# plt.title("Prediction")
# plt.show()