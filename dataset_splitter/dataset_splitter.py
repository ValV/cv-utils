#!/usr/bin/python

import os
import string
import cv2 as cv
import numpy as np
import argparse

from sys import argv
#from os import (access, makedirs, path, walk, R_OK)
from shutil import copyfile

class ImageDirectory(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    dataset_path = values
    #print(dataset_path)
    if not os.path.isdir(dataset_path[0]):
      raise argparse.ArgumentTypeError("Path {0} is not a valid directory"\
          .format(dataset_path[0]))
    if os.access(dataset_path[0], os.R_OK):
      setattr(namespace, self.dest, dataset_path[0])
    else:
      raise argparse.ArgumentTypeError("Path {0} is not readable directory"\
          .format(dataset_path[0]))

def image_move(filename: str, key: int):
  source = filename
  label = chr(key).lower()
  destination = os.path.join(label, os.path.basename(filename))
  #print("Original filename: {0}".format(source))
  try:
    os.makedirs(label, exist_ok=True)
    copyfile(source, destination)
  except IOError as e:
    print("Error copying to {1}:\n{0}".format(e, destination))
  else:
    print("Image has been copied to {0}".format(destination))
  return None

def fit_svga(image: np.ndarray):
  (svga_height, svga_width) = (600, 800)
  (height, width) = image.shape[:2]
  #print("Original image size: {0}, {1}".format(width, height))
  ratio_height = svga_height / float(height)
  ratio_width = svga_width / float(width)
  ratio = min(ratio_height, ratio_width)
  size = (round(width * ratio), round(height * ratio))
  #print("Target image size: {0}x{1}".format(size[0], size[1]))
  return cv.resize(image, size, cv.INTER_AREA)

parser = argparse.ArgumentParser(description="Imagery dataset splitter")
parser.add_argument("path", nargs=1, action=ImageDirectory, 
    help="Image dataset directory")

try:
  args = parser.parse_args()
  image_files = []
  high = range(ord('A'), ord('Z') + 1)
  low = range(ord('a'), ord('z') + 1)
  image_info = [
      "<a>...<z>: copy image to the corresponding directories a...z",
      "<Space>: skip the image to the next cycle (if not sure)",
      "<Esc>: exit program",
  ]

  #print(args.path)
  for root, dirs, files in os.walk(args.path):
    #print(files)
    for filename in files:
      image_files.append(os.path.join(root, filename))
  #print(type(images), images)
  cycle = 0
  while len(image_files):
    remain = []
    images_total = len(image_files)
    images_labeled = 0
    images_skipped = len(remain)
    print("Processing cycle #{0:4d}:".format(cycle))
    
    for image_file in image_files:
      #print(image_file)
      image = cv.imread(image_file, cv.IMREAD_COLOR)
      if type(image) is np.ndarray:
        # Process image file
        image = fit_svga(image)
        for i, line in enumerate(image_info):
          cv.putText(image, line, (32, 32 + i * 16),
              cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
          cv.putText(image, line, (32, 32 + i * 16),
              cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        image_paths = "Filename: {0}".format(image_file)
        cv.putText(image, image_paths,
            (32, image.shape[0] - 32), cv.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 0), 2)
        cv.putText(image, image_paths,
            (32, image.shape[0] - 32), cv.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1)
        
        image_stats = "Images total/labeled/skipped: {0}/{1}/{2}"\
            .format(images_total, images_labeled, images_skipped)
        cv.putText(image, image_stats,
            (32, image.shape[0] - 48), cv.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 0), 2)
        cv.putText(image, image_stats,
            (32, image.shape[0] - 48), cv.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1)
        
        #cv.startWindowThread()
        cv.namedWindow(image_file, cv.WINDOW_AUTOSIZE)
        cv.moveWindow(image_file, 600 - image.shape[1] // 2,
                                  384 - image.shape[0] // 2)
        cv.imshow(image_file, image)
        while True:
          key = cv.waitKey()
          if key in low or key in high:
            # A..Z: move the image into a..z directories
            image_move(image_file, key)
            images_labeled += 1
            cv.destroyWindow(image_file)
            break
          elif key == 32:
            # Space: skip into the next cycle
            print("Skipping {0} to the next cycle.".format(image_file))
            remain.append(image_file)
            images_skipped += 1
            cv.destroyWindow(image_file)
            break
          elif key == 27:
            # Esc: exit program
            print("Imagery processing aborted...")
            parser.exit()
            break
      # else: skip image file
    image_files = remain
    cycle += 1
  print("All imagery was successfully splitted!")
  cv.destroyAllWindows()
except argparse.ArgumentError as e:
  print("Argument:", e)
except IsADirectoryError as e:
  print("File:", e)
except OSError as e:
  print("OS:", e)

# vim: se et sts=2 sw=2 number:
