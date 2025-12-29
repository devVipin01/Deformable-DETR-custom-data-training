import os
import random
import numpy as np
from PIL import Image
import cv2 as cv
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
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
TEST_DIR = os.path.join(dataset, "val2017")

# =========================
# Augmentations
# =========================
def create_augmentations(height=800, width=1333):
    return A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.GaussianBlur(p=0.2),
            A.CLAHE(p=0.2),
            A.HueSaturationValue(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.0,
        ),
    )

def augment_data(image, target, augmentations):
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    bboxes, category_ids = [], []

    for ann in target:
        x, y, bw, bh = ann["bbox"]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        bboxes.append([x, y, bw, bh])
        category_ids.append(ann["category_id"])

    if len(bboxes) == 0:
        return image, target

    augmented = augmentations(
        image=image_np, bboxes=bboxes, category_ids=category_ids
    )

    augmented_image = Image.fromarray(augmented["image"])
    for i, bbox in enumerate(augmented["bboxes"]):
        target[i]["bbox"] = list(bbox)

    return augmented_image, target

# =========================
# Dataset
# =========================
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_dir_path, processor, train=True):
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

        target_dict = {"image_id": image_id, "annotations": target}
        encoding = self.processor(images=img, annotations=target_dict, return_tensors="pt")

        return encoding["pixel_values"].squeeze(0), encoding["labels"][0]

# =========================
# Processor
# =========================
CHECK_POINT = "facebook/deformable-detr-detic"

processor = DeformableDetrImageProcessor.from_pretrained(
    CHECK_POINT,
    size={"shortest_edge": 800, "longest_edge": 1333},
)

train_dataset = CocoDetection(TRAIN_DIR, processor, train=True)
val_dataset = CocoDetection(VAL_DIR, processor, train=False)
test_dataset = CocoDetection(TEST_DIR, processor, train=False)

# =========================
# DataLoader
# =========================
def collate_fn(batch):
    pixel_values = [b[0] for b in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [b[1] for b in batch]

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, collate_fn=collate_fn
)

# =========================
# Model
# =========================
categories = test_dataset.coco.cats
id2label = {k: v["name"] for k, v in categories.items()}

class DeformableDetr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.model = DeformableDetrForObjectDetection.from_pretrained(
            CHECK_POINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch):
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=labels,
        )
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        self.log("validation_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n],
                "lr": self.hparams.lr_backbone,
            },
        ]
        return torch.optim.AdamW(
            param_dicts,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    # ✅ KEEP THESE (THIS IS WHY IT WORKS)
    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

# =========================
# Checkpoints
# =========================
from pytorch_lightning.callbacks import ModelCheckpoint

best_last_ckpt = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-{epoch:02d}-{validation_loss:.4f}",
    monitor="validation_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
)

every_5_epoch_ckpt = ModelCheckpoint(
    dirpath="checkpoints",
    filename="epoch-{epoch:02d}",
    every_n_epochs=5,
    save_top_k=-1,
)

# =========================
# Train
# =========================
model = DeformableDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
summary(model, col_names=["trainable"])

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=10,
    accumulate_grad_batches=4,
    log_every_n_steps=20,
    callbacks=[best_last_ckpt, every_5_epoch_ckpt],
)

trainer.fit(model)

# =========================
# Inference (unchanged)
# =========================
image_id = random.choice(test_dataset.coco.getImgIds())
image_info = test_dataset.coco.loadImgs(image_id)[0]
image_path = os.path.join(test_dataset.root, image_info["file_name"])
image = cv.imread(image_path)

model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.shape[:2]]).to(model.device)
    results = processor.post_process_object_detection(
        outputs, threshold=0.085, target_sizes=target_sizes
    )[0]

detections = sv.Detections.from_transformers(results)
box_annotator = sv.BoxAnnotator()
labels = [f"{id2label[c]} {s:.2f}" for _, s, c, _ in detections]
annotated = box_annotator.annotate(image.copy(), detections, labels)

plt.figure(figsize=(16, 12))
plt.imshow(cv.cvtColor(annotated, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
