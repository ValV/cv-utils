# GeoTIFF convert script

## Description

This script converts 1-band (grayscale) GeoTIFF to 3-band (RGB) GeoTIFF using GDAL library. For now only grayscale -> RGB is implemented.

## Usage

Invoke the script as:

```shell
python gtiff_convert.py <input_file> <output_file>
```

Where `input_file` corresponds to 1-band grayscale GeoTIFF (must exist, obviously), whereas `output_file` is the 3-band RGB GeoTIFF to be created.

## TODO

- [ ] convert to Python module;
- [ ] add more conversion schemes.
