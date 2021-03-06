import os
import argparse

from typing import List
from osgeo.gdal import GDT_UInt16, Warp, WarpOptions


def gdal_callback(completed: float, message: str, args: None) -> bool:
    progress = int(completed * 100)
    if progress == 100:
        print(progress, '- done.')
    elif progress % 10 == 0:
        print(progress, end='', flush=True)
    elif progress % 2 == 0:
        print('.', end='', flush=True)
    return True


def warp(files: List[str], shape: str = None, output: str = 'output') -> bool:
    # Check shapefile
    try:
        if os.path.isfile(shape):
            shape = os.path.abspath(os.path.realpath(shape))
        else:
            raise FileNotFoundError
    except (TypeError, FileNotFoundError) as e:
        print(f"Shapefile '{shape}' does not exist!")
        shape = None
        return False # comment out to continue without shapefile

    # Create GDAL Warp options object
    options = WarpOptions(format='GTiff', dstSRS='EPSG:32640',
                          outputType=GDT_UInt16, geoloc=False,
                          cutlineDSName=shape,
                          cropToCutline=(True if shape else False),
                          callback=gdal_callback)

    # Create output directory
    try:
        output = output or 'output'
        os.makedirs(output, exist_ok=True)
        print(f"Using output directory '{output}'")
    except PermissionError as e:
        print(f"Failed creating output directory {e}!")
        return False

    # Process files
    print(f"Start processing {len(files)} files:")
    for index, filename in enumerate(files):
        try:
            source = os.path.abspath(filename)
            # Skip to next file if this one does not exist
            if not os.path.isfile(source):
                print(f"{index:5d}. Bad file '{source}'!")
                continue
            else:
                print(f"{index:5d}. Processing '{source}'...")
            name, extension = os.path.splitext(os.path.basename(source))
            destination = os.path.join(os.path.abspath(output),
                                       f"{name}_shaped{extension}")
            # Call GDAL Warp
            Warp(destination, source, options=options)
        except Exception as e:
            print(e)

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GDAL Warp Script')
    parser.add_argument('-s', '--shape', nargs='?', metavar='file',
                        help='shapefile cutline')
    parser.add_argument('-o', '--output', nargs='?', metavar='path',
                        help='output directory')
    parser.add_argument('file', nargs='*',
                        help='GeoTIFF file to warp and cut')

    args = parser.parse_args()
    #print(f"Arguments: {args}")
    warp(args.file, args.shape, args.output)

# vim: se et sw=4 sts=4 syntax=python:
