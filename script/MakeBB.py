#!/usr/bin/python3
import argparse
from os import listdir
import os.path
from PIL import Image, ImageDraw #NB : Requires Pillow : https://pillow.readthedocs.io/en/3.1.x/installation.html
import image_manips


# Taken from https://stackoverflow.com/a/14117511/5128743
def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

# Parse Arguments
parser = argparse.ArgumentParser(description="Make a bounding box style mask out of a mask")
parser.add_argument("target", type=str, help="Path (with filename and extension) of the mask from which to make a bounding box")
parser.add_argument("output", type=str, help="Path, name and extension of where to save the output image")
parser.add_argument("-i", dest="image", type=str, default=None, help="Asks for the resulting mask to be applied upon an image, given in here as a path with filename and extension.")
parser.add_argument("-p", dest="padding", type=check_positive, default=0, help="How much padding to add around the minimal bounding box")
parser.add_argument("-r", dest="resize", type=check_positive, default=1, help="Factor to reduce-resize the image by (2 means width/2 and height/2)")
parser.add_argument("--directory", dest="directory", action="store_true", required=False, help="Indicate that the target path and the output path (and the -i image path) is towards a directory of targets and one of outputs instead (and for -i a directory of images associated to the mask by default sort order). For outputs, will use same name as input.")
args = parser.parse_args()

# Prepare list of masks (and images if -i called) to iterate over, and output file names
if args.directory:
    target_names = listdir(args.target)
    target_names.sort()

    output_names = []
    for i, value in enumerate(target_names):
        output_names.append(os.path.join(args.output, os.path.basename(value)))
        target_names[i] = os.path.join(args.target, value)

    if args.image is not None:
        image_names = listdir(args.image)
        image_names.sort()

        for i, value in enumerate(image_names):
            image_names[i] = os.path.join(args.image, value)

else:
    target_names = [args.target]
    if args.image is not None:
        image_names = [args.image]
    output_names = [args.output]

for i, target_name in enumerate(target_names):
    # Pillow image object initialisation
    target = Image.open(target_name)
    original_target = target.copy()
    if args.resize != 1:
        oldTarget = target
        target = image_manips.resize(target, args.resize)
        oldTarget.close()

    target = image_manips.getBoundingBox(target, args.padding)

    if args.image is not None:
        # Apply mask upon the -i image and save result
        input_img = Image.open(image_names[i])
        if args.resize != 1:
            oldinput = input_img
            input_img = image_manips.resize(input_img, args.resize)
            oldinput.close()
        applied_mask = image_manips.applyMask(input_img, target)
        original_target = original_target.resize(applied_mask.size, Image.BICUBIC)
        applied_mask, original_target = image_manips.cropBlack(applied_mask, original_target)
        applied_mask.save(os.path.join(os.path.dirname(output_names[i]), "applied_" + os.path.basename(output_names[i])))
        applied_mask.close()
        input_img.close()
        original_target.save(output_names[i])

    else:
        # Save output
        target.save(output_names[i])
    target.close()
    original_target.close()
