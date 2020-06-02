# GDAL Warp Script

## Example usage:

```shell
python warp.py -s shapefile.shp *.tif
```

This will process all the `*.tif` files in current directory and write output files into `output/` directory by default.

> Without `-s shapefile.shp` provided the script will stop (see source to fix it).

## Full help:

```
usage: warp.py [-h] [-s [file]] [-o [path]] [file [file ...]]

GDAL Warp Script

positional arguments:
  file                  GeoTIFF file to warp and cut

optional arguments:
  -h, --help            show this help message and exit
  -s [file], --shape [file]
                        shapefile cutline
  -o [path], --output [path]
                        output directory
```
