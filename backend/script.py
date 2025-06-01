image_path=input()
prompt=input()

import os
HOME = os.getcwd()
print("HOME:", HOME)

GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

SAM_CHECKPOINT_PATH = os.path.join("/app/weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

import sys
dino_path = r"/app/GroundingDINO/"
sys.path.append(lama_path)

from groundingdino.util.inference import Model

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
SAM_ENCODER_VERSION = "vit_h"
DEVICE = "cuda"

from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

import numpy as np
import supervision as sv

from groundingdino.util.inference import load_model, load_image, predict, annotate
model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)

IMAGE_NAME = "dog-4.jpeg"
IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)

TEXT_PROMPT = "bag"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

# Convert GroundingDINO boxes to NumPy array
xyxy = boxes.cpu().numpy()  # Convert tensor to numpy array
confidence = logits.cpu().numpy()  # Convert logits to numpy array

# Create a class_id mapping (optional)
class_id_map = {phrase: idx for idx, phrase in enumerate(set(phrases))}
class_ids = np.array([class_id_map[phrase] for phrase in phrases], dtype=object)

# Create a Supervision Detections object
detections2 = sv.Detections(
    xyxy=xyxy,
    confidence=confidence,
    class_id=class_ids
)

# Annotate the image
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# Display the annotated image
%matplotlib inline
#sv.plot_image(annotated_frame, (16, 16))

# Print the detections
print(detections)


import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2

device = "cuda"  # or "cpu"

# Set the image for SAM
sam_predictor.set_image(image_source)

# Get image dimensions
H, W, _ = image_source.shape

# Convert GroundingDINO boxes to SAM input format
boxes_scaled = boxes.clone().cpu()

# GroundingDINO gives [center_x, center_y, width, height]
# Convert to [x1, y1, x2, y2] format
boxes_xyxy = torch.zeros_like(boxes_scaled)
boxes_xyxy[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2  # x1 = center_x - width/2
boxes_xyxy[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2  # y1 = center_y - height/2
boxes_xyxy[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2  # x2 = center_x + width/2
boxes_xyxy[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2  # y2 = center_y + height/2

# Scale the normalized coordinates to pixel coordinates
boxes_xyxy[:, [0, 2]] *= W  # scale x coordinates
boxes_xyxy[:, [1, 3]] *= H  # scale y coordinates

# Convert to numpy array
boxes_scaled = boxes_xyxy.numpy()

# Print the coordinates for debugging
print("Original normalized coordinates:", boxes.cpu().numpy())
print("Converted to [x1,y1,x2,y2] format:", boxes_xyxy.numpy())
print("Final scaled pixel coordinates:", boxes_scaled)

# Get masks for all detected boxes
masks_list = []
scores_list = []
for box in boxes_scaled:
    # Convert box to format expected by SAM
    sam_box = box.astype(int)

    # Get mask prediction from SAM
    masks, scores, logits = sam_predictor.predict(
        box=sam_box,
        multimask_output=False
    )

    masks_list.append(masks[0])
    scores_list.append(scores[0])

# Convert masks to a single numpy array
masks = np.stack(masks_list, axis=0)

# Visualization
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              edgecolor='red', facecolor=(0,0,0,0), lw=2))

# Create an empty combined mask (same size as individual masks)
combined_mask = np.zeros_like(masks[0], dtype=np.uint8)

# Apply bitwise OR for each mask
for mask in masks:
    mask_uint8 = (mask.astype(np.uint8)) * 255  # Convert boolean to uint8 (255 for True, 0 for False)
    combined_mask = cv2.bitwise_or(combined_mask, mask_uint8)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 10))

# **Left: Original Image with Bounding Boxes**
axes[0].imshow(image_source)
for box in boxes_scaled:
    show_box(box, axes[0])  # Draw bounding boxes
axes[0].axis("off")
axes[0].set_title("Original Image with Bounding Boxes")

# **Right: Combined Mask**
axes[1].imshow(combined_mask, cmap="gray")
axes[1].axis("off")
axes[1].set_title("Combined Mask")

# Show the figure
# plt.show()

cv2.imwrite("combined_mask.png", combined_mask)