from sys import argv
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
object_min = 200
# Bump map thresholding values
thresh_low = 3
thresh_high = 255
# Epsilon value for object's curve polygonal approximation
e = m.pi**2
# Output image maximum width
out_max_width = 800

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

# Process command-line. Initialize argument parser
parser = argparse.ArgumentParser(description="Segmentation script for sonar images")
parser.add_argument('files', nargs='+', type=argparse.FileType('rb'), help='File names to process.')
# Argument mutually exclusive group: either bounding box or convex hull
parser.add_mutually_exclusive_group()
parser.add_argument('-b', '--bbox', action='store_true', help='Draw bounding box around objects (excludes --hull).')
parser.add_argument('-u', '--hull', action='store_true', help='Draw convex hull around objects (excludes --bbox).')
# Parse or die!
try:
  args = parser.parse_args(argv)
except argparse.ArgumentError as e:
  print("Error:", e)
except IndexError as e:
  # This error occurs while processing ArgumentError
  # Use exception chaining to pop up the real reason
  print("Error:", e.__cause__ or e.__context__ or e)
else:
  for sonar_file in args.files[1:]:
    with sonar_file:
      img = np.frombuffer(sonar_file.read(), np.uint8)
    org = cv.imdecode(img, cv.IMREAD_COLOR)
    img = cv.cvtColor(org, cv.COLOR_BGR2RGB)
    # Make all the given  pixels black
    #quench(img, quench_colors)
    #img[np.where((img!=bump_colors[-1]).all(axis=2))] = [0,0,0]
    # Display color set
    #np.set_printoptions(formatter={'int':lambda x:hex(int(x))})
    #print("Color set:", colorset(img))
    # Perform reflection -> volume transformation (bump mapping)
    img = bump(img, bump_colors)
    # Erase top region
    img[:erase_top,:,] = np.zeros((1,3)) # erase top n lines (px)
    # Image filtering
    imgray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(imgray, thresh_low, thresh_high, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # Preform distance and morphology transformations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Find sure background
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Find sure foreground area
    #dist_trans = cv.distanceTransform(opening, cv.DIST_L2, 5)
    #_, sure_fg = cv.threshold(dist_trans, 0.7 * dist_trans.max(), 255, 0)

    # Unknown regions
    #sure_fg = np.uint8(sure_fg)
    #unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    #_, markers = cv.connectedComponents(sure_fg)

    # Increment labels for background to be 1
    #markers = markers + 1

    # Set unknown region as zero
    #markers[unknown==255] = 0

    # Apply watershed
    #markers = cv.watershed(img, markers)
    #img[markers==-1] = [255, 0, 0]

    # Find contours
    contours, hierarchy = cv.findContours(sure_bg, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)

    # Process contours
    areas = []
    hulls = []
    rects = []
    for contour in contours:
      if (cv.contourArea(contour)) > object_min:
        epsilon = e * m.erf(cv.arcLength(contour, True) / (m.pi * object_min))
        areas.append(cv.approxPolyDP(contour, epsilon, True))
        hulls.append(cv.convexHull(contour, False))
        rects.append(cv.boundingRect(contour))
    if args.bbox:
      #print("Bounding box mode.")
      for r in rects:
        cv.rectangle(org, r[:2], (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)
    elif args.hull:
      #print("Convex hull mode.")
      cv.drawContours(org, hulls, -1, (255, 0, 0), 2)
    else:
      #print("Primary contours mode.")
      cv.drawContours(org, areas, -1, (0, 255, 0), 2)

    # Display results (OpenCV + Matplotlib versions)
    h, w = org.shape[:2]
    scale = out_max_width / w
    res = cv.resize(org, (out_max_width, round(h * scale)))
    cv.imshow(argv[1], res)
    # Wait for Delete (255 or 127), Esc (27) or Enter (13)
    while cv.waitKey() not in (255, 127, 27, 13):
      continue
    cv.destroyAllWindows()
    #plt.imshow(img)
    #plt.show()
finally:
  parser.exit()

# vim: se et ts=2 sw=2 number:
