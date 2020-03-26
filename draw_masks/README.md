# Draw Masks

## Description

Apply masks to the SAR HH and HV ploarized images. Used to draw predicted masks on Tellus SAR data (from Signate competition).

## Usage

Ensure you have folders `test_images` and `sar-rgb-mux-0.0084` in your working directory with the contents like:

* test images
```
test_images
├── test_hh_00.jpg
├── test_hv_00.jpg
...
├── test_hh_39.jpg
└── test_hv_39.jpg
```

* predicted masks
```
sar-rgb-mux-0.0084
├── test_00.png
...
└── test_39.png
```

Then run:
```
python draw_masks.py
```

Easy...
