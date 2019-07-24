#!/usr/bin/python

from sys import argv
from os import makedirs
from shutil import copy
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import numpy as np
import math as m
import cv2 as cv
import argparse

# Bump colors represent reflection from weakest (upper values)
# to strongest (lower values). This is the coolest feature
bump_colors = [
    [0xff,0xff,0xff],
    [0x9f,0x9f,0x9f],
    [0x5f,0x5f,0x5f],
    [0x00,0x00,0xff],
    [0x00,0x00,0x7f],
    [0x00,0xbf,0x00],
    [0x00,0x7f,0x00],
    [0xff,0xff,0x00],
    [0xff,0x7f,0x00],
    [0xff,0x00,0xbf],
    [0xff,0x00,0x00],
    [0xa6,0x53,0x3c],
    [0x78,0x3c,0x28]]
# List of colors to quench (turn into black)
quench_colors = [[95, 95, 95],[159, 159, 159],[255, 255, 255]]
# Number of pixels to remove from the top
erase_top = 32
# Object's minimum area (by shoelace algorithm)
object_min = 440
# Bump map thresholding values
thresh_low = 255 - round(2 * 255 / len(bump_colors))
thresh_high = 255
# Epsilon value for object's curve polygonal approximation
eps = m.pi**2
# Output image maximum width
out_max_width = 800
# Output folder
out_dir: str = 'data'
dst_dir: str = out_dir + '/images/img'
jsn_dir: str = out_dir + '/images/json'

# Function to detect image's color set (all possible color values)
def colorset(img):
  return np.unique(img.reshape(-1, img.shape[2]), axis=0)

# Function to transform initial reflection colors into gray bump map
def bump(img, colors):
  n = len(colors)
  new = np.full((1, 3), 255 / n)
  for i in range(0, n):
    img[np.where((img==colors[i]).all(axis=2))] = new * i
  return img

# Function to quench certain colors (turn into black)
def quench(img, colors):
  for color in colors:
    img[np.where((img==color).all(axis=2))] = [0, 0, 0]
  return img

# Function to erase top num lines
def remove_top(img, num):
  img[:num,:,] = np.zeros((1,3)) # (px)
  return img

# Function to remove black [0, 0, 0] lines from side to side
def remove_grid(img):
  img = img[2:-2,2:-2,:] # trim borders
  mask = np.all(img, axis=2)
  cols = np.any(mask, axis=0)
  rows = np.any(mask, axis=1)
  img = img[rows, :, :]
  img = img[:, cols, :]
  return img

# Function to draw contours/hulls/boxes on a given image
def draw_contours(img, contours):
  if args.bbox:
    #print("Bounding box mode.")
    for r in contours:
      cv.rectangle(img, r[:2], (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)
  elif args.hull:
    #print("Convex hull mode.")
    cv.drawContours(img, contours, -1, (255, 0, 0), 2)
  else:
    #print("Primary contours mode.")
    cv.drawContours(img, contours, -1, (0, 255, 0), 2)
  #cv.imshow("DEBUG", img)
  #cv.waitKey()
  return img

# Function to get bounding boxes from contours
def get_rects(contours):
  rects = []
  for contour in contours:
    rects.append(cv.boundingRect(contour))
  return rects

# Function to get convex hulls from contours
def get_hulls(contours):
  hulls = []
  for contour in contours:
    hulls.append(cv.convexHull(contour, False))
  return hulls

# Function to get seabed edge mask
def get_edge(grayscale):
  mask = np.rot90(np.array(grayscale), 3)
  for row in range(0, mask.shape[0]):
    edge = True
    for col in range(0, mask.shape[1] - 1):
      if (mask[row, col] >= 230) and (mask[row, col + 1] < mask[row, col]):
        edge = False
      if edge:
        mask[row, col] = 0
      else:
        mask[row, col] = 255
  mask = shift(mask, (0, 5), cval=0) # TODO: make configurable option
  mask = np.rot90(mask)
  return mask

# Function to get contours from the image
def get_contours(img):
  contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
  areas = []
  for contour in contours:
    # Skip contours with area below minimal (very small objects)
    if (cv.contourArea(contour)) < object_min:
      continue
    # Approximate and add a contour
    epsilon = eps * m.erf(cv.arcLength(contour, True) / (m.pi * object_min))
    areas.append(cv.approxPolyDP(contour, epsilon, True))
  return areas

# Process command-line. Initialize argument parser
parser = argparse.ArgumentParser(description="Segmentation script for sonar images")
parser.add_argument('file', nargs='+', type=argparse.FileType('rb'), help='File name to process.')
# Argument mutually exclusive group: either bounding box or convex hull
mutex_group = parser.add_mutually_exclusive_group()
mutex_group.add_argument('-b', '--bbox', action='store_true', help='Draw bounding box around objects (excludes --hull).')
mutex_group.add_argument('-u', '--hull', action='store_true', help='Draw convex hull around objects (excludes --bbox).')
# Parse or die!
try:
  args = parser.parse_args()
  makedirs(dst_dir, exist_ok=True)
  makedirs(jsn_dir, exist_ok=True)

  # Process files
  for sonar_file in args.file[1:]:
    with sonar_file:
      img_name = sonar_file.name
      img = np.frombuffer(sonar_file.read(), np.uint8)
    org = cv.imdecode(img, cv.IMREAD_COLOR)

    org = remove_grid(org) # breaks image (needs at least one conversion)
    img = cv.cvtColor(org, cv.COLOR_BGR2RGB)
    org = cv.cvtColor(img, cv.COLOR_RGB2BGR) # fix broken origin image

    # Perform reflection -> volume transformation (bump mapping)
    img = bump(img, bump_colors)

    # Erase top region
    img = remove_top(img, erase_top)

    # Image filtering
    imgray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    seabed_mask = get_edge(imgray) # this is too slow (TODO: optimize)
    #print("Seabed mean:", np.mean(seabed_mask))

    kernel_size = int(m.e**2)#round(m.sqrt(object_min)) - 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernelx = np.ones((4, 4), np.uint8)

    seabed_mask = cv.GaussianBlur(seabed_mask, (3, 3), 0)

    # Extract seabed from the image
    seabed = None
    seabed_contours = None
    if np.mean(seabed_mask) > 127:
      seabed = cv.bitwise_and(imgray, 255 - seabed_mask)
      _, seabed = cv.threshold(seabed, 200, thresh_high, cv.THRESH_BINARY)
      #seabed = cv.morphologyEx(seabed, cv.MORPH_OPEN, kernel, iterations=2)
      seabed = cv.morphologyEx(seabed, cv.MORPH_CLOSE, kernel, iterations=2)
      seabed_contours = get_contours(seabed)
      #plt.imshow(seabed, cmap='gray', vmin=0, vmax=255)
      #plt.show()

    # Extract fish from the image
    fishes = None
    fishes_contours = None
    if not seabed is None:
      fishes = cv.bitwise_and(imgray, seabed_mask)
    else:
      fishes = imgray
    fishes = cv.GaussianBlur(fishes, (kernel_size, kernel_size), 0)
    _, fishes = cv.threshold(fishes, thresh_low, thresh_high, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Preform distance and morphology transformations
    fishes = cv.morphologyEx(fishes, cv.MORPH_OPEN, kernelx, iterations=2)
    fishes = cv.morphologyEx(fishes, cv.MORPH_CLOSE, kernel, iterations=2)

    # Draw contours
    fishes_contours = get_contours(fishes)
    if args.bbox:
      contours = get_rects(fishes_contours)
    elif args.hull:
      contours = get_hulls(fishes_contours)
    else:
      contours = fishes_contours
    org = draw_contours(org, contours)

    # Display results (OpenCV + Matplotlib versions)
    h, w = org.shape[:2]
    scale = out_max_width / w
    res = cv.resize(org, (out_max_width, round(h * scale)))
    cv.imshow(img_name, res)
    # Wait for Delete (255 or 127), Space(32), Esc (27) or Enter (13)
    while True:
      key = cv.waitKey()
      if key in (255, 127):
        # Remove file from collection if Delete is pressed (TODO)
        break
      elif key == 32:
        # Skip the image if Space is pressed
        break
      elif key == 27:
        # Exit program if Esc is pressed
        parser.exit()
        break
      elif key == 13:
        # Save the file to destination if Enter is pressed
        try:
          copy(img_name, dst_dir)
        except OSError as e:
          print("Error:", e)
        break
    cv.destroyAllWindows()
except argparse.ArgumentError as e:
  print("Error:", e)
except IndexError as e:
  # This error occurs while processing ArgumentError
  # Use exception chaining to pop up the real reason
  print("Error:", e.__cause__ or e.__context__ or e)
except OSError as e:
  print("Error:", e)

# vim: se et ts=2 sw=2 number:
