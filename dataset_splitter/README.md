# Image Dataset Splitter

## Description

This script automates splitting a set of images into subsets with alphabetical labels from `a` to `z`. This script requires OpenCV to be installed.

The script copies source images cyclically, i.e. one can skip certain images from being labeled (and copied under the label directory) at once.

## Usage

The script requires a directory with images as input:

```shell
$ python dataset_splitter.py path/to/images/
```

## Dataset

The dataset is a directory with images readable by OpenCV. Source dataset is passed as an argument to the script. Output splitted dataseet is created in the current (working) directory.

## Key bindings

* `Esc` - terminate execution;
* `Space` - skip to the next image;
* `a`...`z` - copy current image to the directory labelled by the key.

## TODO

- [ ] multiple input directories/datasets;
- [ ] format code.
