import argparse
import os
import sys
import cv2
import numpy as np
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model, load_model, load_image, predict, annotate

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run SAM with GroundingDINO")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Text prompt for detection")
    parser.add_argument("--output_path", required=True, help="Path to save output mask")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
    return parser.parse_args(args)

def main(args=None):
    args = parse_args(args)
    HOME = os.getcwd()
    
    # Setup paths
    GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
    SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    
    # Initialize models
    model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)
    SAM_ENCODER_VERSION = "vit_h"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # Load and process image
    image_source, image = load_image(args.image_path)
    
    # Run GroundingDINO
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    # Process with SAM
    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    
    boxes_scaled = boxes.clone().cpu()
    boxes_xyxy = torch.zeros_like(boxes_scaled)
    boxes_xyxy[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2
    
    boxes_xyxy[:, [0, 2]] *= W
    boxes_xyxy[:, [1, 3]] *= H
    boxes_scaled = boxes_xyxy.numpy()

    # Get masks
    masks_list = []
    for box in boxes_scaled:
        sam_box = box.astype(int)
        masks, scores, logits = sam_predictor.predict(
            box=sam_box,
            multimask_output=False
        )
        masks_list.append(masks[0])

    # Combine masks
    masks = np.stack(masks_list, axis=0)
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    
    for mask in masks:
        mask_uint8 = (mask.astype(np.uint8)) * 255
        combined_mask = cv2.bitwise_or(combined_mask, mask_uint8)

    # Save result
    cv2.imwrite(args.output_path, combined_mask)
    return "Success"

if __name__ == "__main__":
    main()