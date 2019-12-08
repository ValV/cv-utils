from os import path
from sys import (argv, exit)
from osgeo import gdal

def print_usage():
    print("Program for converting grayscale to RGB GeoTIFF using GDAL")
    print(f"{path.basename(argv[0])} <input GRAYSCALE> <output RGB>")

def convert_gray2rgb(source, destination):
    options = gdal.TranslateOptions(
            format="GTiff",
            outputType=gdal.GDT_Byte,
            bandList=["1,1", "1,2", "1,3"],
            creationOptions=["PHOTOMETRIC=RGB"])
    gdal.Translate(destination, source, options=options)

def main():
    if len(argv) != 3 or not path.exists(argv[1]):
        print_usage()
        exit(1)

    print("Grayscale:", argv[1])
    print("RGB:", argv[2])
    print("Converting from grayscale to RGB...")

    convert_gray2rgb(argv[1], argv[2])
    print("Done!")

if __name__ == '__main__':
    main()

# vim: se et sts=4 sw=4 number:
