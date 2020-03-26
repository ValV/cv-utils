import os
import cv2 as cv
import numpy as np

from glob import glob

# Input data
image_path = "test_images"
mask_path = "sar-rgb-mux-0.0084"
image_files = sorted(glob(os.path.join(image_path, "*")))
mask_files = sorted(glob(os.path.join(mask_path, "*")))

# Output data
output_path = "output_images"
os.makedirs(output_path, exist_ok=True)

#print("Images:\n", "\n".join(image_files), sep="")
#print("Masks:\n", "\n".join(mask_files), sep="")

for image_file in image_files:
    # print(f"Image: {os.path.basename(image_file)}")
    image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
    mask = cv.imread(image_file.replace(image_path, mask_path)\
                     .replace("_hh", "").replace("_hv", "")\
                     .replace("jpg", "png"), cv.IMREAD_GRAYSCALE)
    height, width = [int(x / 2) for x in image.shape[:2]]
    assert type(image) is np.ndarray and type(mask) is np.ndarray, \
    "Failed to read image and mask"
    # print(f"Resizing...")
    image = cv.resize(image, (width, height))
    mask = cv.resize(mask, (width, height))
    print(f"Image: {os.path.basename(image_file)}, size = {image.shape[:2]}")
    # print(f"Equalizing...")
    image = cv.equalizeHist(image)
    # print(f"Creating output and overlay...")
    output = np.repeat(image[:,:,np.newaxis], 3, axis=2)
    overlay = output.copy()
    # print(f"Applying mask...")
    overlay[...,0] *= (mask * -1 + 1).astype(np.uint8)
    overlay[...,1] *= (mask * -1 + 1).astype(np.uint8)
    overlay[...,2] |= (mask * 255).astype(np.uint8)
    # print(f"Merging output with overlay...")
    output = cv.addWeighted(output, 0.85, overlay, 0.15, 0)
    # print(f"Saving {type(output)}...")
    cv.imwrite(os.path.join(output_path, os.path.basename(image_file)), output)
    # break
