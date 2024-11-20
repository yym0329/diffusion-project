# 영민님 code base를 parallel하게 돌리기 위함
import json
import numpy as np
from PIL import Image
import random
import os
import glob
import argparse
from dataclasses import dataclass
from typing import Optional
import imageio
import numpy as np
import cv2
import simple_parsing
from transformers import pipeline

from typing import Dict



# Make jsonl file, which contains the path to the images and masks
# TODO 이거 개조 필요
# 내쪽에서 train split이랑 test split으로 함수 구성 필요
def make_jsonl(output_path, data_dir):
    """
    Make a jsonl file that contains the path to the images and masks.
    The jsonl file should have the following format:

    {"image_path": "path/to/image", "mask_path": "path/to/mask"}
    {"image_path": "path/to/image", "mask_path": "path/to/mask"}

    Args:
            output_path (str): path to the output jsonl file.
            data_dir (str): path to the data directory.
    """
    with open(output_path, "w") as f:
        images_dir = os.path.join(data_dir, "images")
        masks_dir = os.path.join(data_dir, "masks")

        for object_name in os.listdir(images_dir):
            for image_name in os.listdir(os.path.join(images_dir, object_name)):
                image_path = os.path.join(images_dir, object_name, image_name)
                mask_name = image_name[:-11] + image_name[-7:]
                mask_path = os.path.join(masks_dir, object_name, mask_name)
                f.write(
                    json.dumps({"image_path": image_path, "mask_path": mask_path})
                    + "\n"
                )


# Use this function to preprocess data
def center_crop_img(tgt_img_path, mask_img_path, resoultion:float = 1.0):
    # DO NOT modify the hyperparameters
    if resoultion == 1.0:
        RESIZE_H, RESIZE_W = 100, 100
        H, W = 128, 128
    elif resoultion == 4.0:
        RESIZE_H, RESIZE_W = 400, 400
        H, W = 512, 512
    else:
        raise ValueError("Invalid resolution value. Please use 1.0 or 4.0")
    
    """
    Preprocess the image and mask to center crop and resize to 128x128

    Args:
            tgt_img_path (str): Path to the target image
            mask_img_path (str): Path to the mask image

    Returns:
            (img_canvas, mask_canvas): Tuple of PIL images
    """
    tgt_img = Image.open(tgt_img_path).convert("RGB")
    np_tgt_img = np.array(tgt_img)

    # mask is processed as [0, 255] value
    mask_img = Image.open(mask_img_path).convert("RGB")  # Foreground mask
    # For some of the masks are given as [0, 255]
    if np.array(mask_img).max() > 1:
        np_mask_img = np.array(mask_img)
    else:
        np_mask_img = np.array(mask_img) * 255
    assert (
        np_mask_img.max() <= 255 and np_mask_img.min() >= 0
    ), f"{np_mask_img.min()}, {np_mask_img.max()}"
    np_tgt_img[np_mask_img == 0] = 255

    # Crop image using bbox
    y, x, r = np.where(np_mask_img == 255)  # Get bbox using the mask
    x1, x2, y1, y2 = x.min(), x.max(), y.min(), y.max()

    crop_img = Image.fromarray(np_tgt_img).crop((x1, y1, x2, y2))
    cropped_mask = Image.fromarray(np_mask_img).crop((x1, y1, x2, y2))
    w = x2 - x1
    assert w > 0, f"{x2} - {x1} = {w}"
    h = y2 - y1
    assert h > 0, f"{y2} - {y1} = {h}"

    # Resize image with respect to max length
    max_length = max(w, h)
    ratio = RESIZE_W / max_length
    resized_w, resized_h = round(w * ratio), round(h * ratio)  # Avoid float error
    assert resized_h == RESIZE_H or resized_w == RESIZE_W

    resized_img = crop_img.resize((resized_w, resized_h))
    resized_object_mask = cropped_mask.resize((resized_w, resized_h))
    img_canvas = Image.new("RGB", (H, W), (255, 255, 255))
    mask_canvas = Image.new("RGB", (H, W), (0, 0, 0))
    pos_w, pos_h = resized_w - W, resized_h - H

    pos_w = abs(pos_w) // 2
    pos_h = abs(pos_h) // 2
    assert pos_w + resized_w <= W and pos_h + resized_h <= H

    img_canvas.paste(resized_img, (pos_w, pos_h))
    mask_canvas.paste(resized_object_mask, (pos_w, pos_h))

    return img_canvas, mask_canvas


def process_image(data_dict: Dict[str, str], output_dir: str, resoultion:float = 1.0):
    """
    crop the object image given the mask. and save the cropped image and mask to the output directory.

    Args:
            k (str): key of the data.
            output_dir (str): output directory.
            data (dict): data dictionary. data[k] = {tgt_img_path: str, mask_path: str},
            where tgt_img_path is the path to the target image and mask_path is the
            path to the mask image.
    """
    # tgt_img_path = data[k]["tgt_img_path"]
    # mask_img_path = data[k]["mask_path"]
    
    k = data_dict["key"]
    tgt_img_path = data_dict["image_path"]
    mask_img_path = data_dict["mask_path"]
    
    img, mask = center_crop_img(tgt_img_path, mask_img_path, resoultion=resoultion)

    img_name = k + ".png"
    # key example: 'obj_28_metal_bucket_010_NA3'
    object_name = k[:-8]
    viewpoint_id = k[-3:]

    image_dir = os.path.join(output_dir, "images", object_name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
    output_img_path = os.path.join(image_dir, img_name)

    mask_dir = os.path.join(output_dir, "masks", object_name)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir, exist_ok=True)

    mask_file_name = object_name + "_" + viewpoint_id + ".png"
    output_mask_path = os.path.join(mask_dir, mask_file_name)

    img.save(output_img_path)
    mask.save(output_mask_path)
    print(f"Saved {output_img_path}")
    print(f"Saved {output_mask_path}")

