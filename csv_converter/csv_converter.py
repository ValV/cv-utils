from sys import argv
from sys import exit
import json
import csv

# Fields' indicies 0..n
classes = 3
description = 4
# Fields' separators
sep_classes = ','
sep_descitems = '\n'

# Split functions for complex data strings
def parse_classes(item):
    return item.replace(' ', '').split(sep_classes)

def parse_description(item):
    items = item.split(sep_descitems)
    features = list()
    for feature in items:
        if len(feature) > 0:
            features.extend(feature.split(". "))
    map(lambda it: it.strip(), features)
    return features

# Command-line arguments
def help():
    print("Usage: python", argv[0], "<in CSV> [out JSON]")

# Process CSV file
if not (len(argv) in range(2, 4)):
    help()
    exit(1)

try:
    jout = list()
    with open(argv[1], newline='') as ifile:
        reader = csv.reader(ifile, delimiter=';')
        header = next(reader)
        data = list(reader)
        for item in data:
            # Split fields into lists by separators
            item[classes] = parse_classes(item[classes])
            item[description] = parse_description(item[description])
            # Zip lists into dicts (JSON objects)
            jout.append(dict(zip(header, item)))
    if len(argv) == 2:
        print(json.dumps(jout, indent=4, ensure_ascii=False))
    elif len(argv) == 3:
        try:
            with open(argv[2], "w+") as ofile:
                json.dump(jout, ofile, indent=4, ensure_ascii=False, skipkeys=True)
        except:
            print("Error writing output file! Make sure the path valid.")
except:
    print("Error processing input file! Make sure CSV is valid.\n")
    help()
