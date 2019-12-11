# OpenCV segmentation script

## Description

This script uses color modifications and image morphology to perform segmentation on certain sonar images.

## Usage

The script requires at least one sonar image as input:

```shell
$ python cv_contours.py path/to/images/image.png
```

It's also possible to pass multiple files:

```shell
$ python cv_contours.py path/to/images/*.png
```

To get help on usage, issue:

```shell
$ python cv_contours.py --help
```

The script can display segmented areas also either as convex hulls or bounding boxes with swithches `-u` and `-b` respectively (see `--help`).

## Collection

This script also intended to save regions of interest as JSON metadata.

> This script automatically creates `data` folder in the working directory path

## Key bindings

* `Esc` - terminate execution;
* `Space` - skip to the next image;
* `Enter` - copy current image to collection.

## TODO

- [ ] collection in Supervisely dataset format (JSON);
- [ ] delete key to remove an image from collection/dataset;
- [ ] command-line options for thresholding and morphology parameters.
