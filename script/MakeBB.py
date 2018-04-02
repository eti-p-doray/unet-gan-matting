#!/usr/bin/python3
import argparse
from os import listdir
import os.path
from PIL import Image, ImageDraw #NB : Requires Pillow : https://pillow.readthedocs.io/en/3.1.x/installation.html

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
parser.add_argument("--directory", dest="directory", action="store_true", required=False, help="Indicate that the target path and the output path (and the -i image path) is towards a directory of targets and one of outputs instead (and for -i a directory of images associated to the mask by default sort order). For outputs, will use same name as input.")
args = parser.parse_args()

# Prepare list of masks (and images if -i called) to iterate over, and output file names
if args.directory:
    target_names = listdir(args.target)
    target_names.sort()

    target_dir = args.target
    if not target_dir.endswith("/"):
        target_dir += "/"

    output_dir = args.output
    if not output_dir.endswith("/"):
        output_dir += "/"

    output_names = []
    for i, value in enumerate(target_names):
        output_names.append(output_dir + os.path.basename(value))
        target_names[i] = target_dir + value

    if args.image is not None:
        image_names = listdir(args.image)
        image_names.sort()

        image_dir = args.image
        if not image_dir.endswith("/"):
            image_dir += "/"

        for i, value in enumerate(image_names):
            image_names[i] = image_dir + value

else:
    target_names = [args.target]
    if args.image is not None:
        image_names = [args.image]
    output_names = [args.output]

for i, target_name in enumerate(target_names):
    # Pillow image object initialisation
    target = Image.open(target_name)
    output = ImageDraw.Draw(target)

    # Get bounding box (bb) around the non-zero pixels of the target mask
    bb = target.getbbox() # returns (Left, Up, Right, Down)

    # Fit bounding box to image coordinates
    bb_coords = [0,0,0,0] # will contain 2 points
    bb_coords[0] = max(0, bb[0] - args.padding) # Point1 x
    bb_coords[1] = max(0, bb[1] - args.padding) # Point1 y
    bb_coords[2] = min(target.size[0], bb[2] + args.padding) # Point2 x
    bb_coords[3] = min(target.size[1], bb[3] + args.padding) # Point2 y

    # Set color to white
    if target.mode == "1": #binary image, 0 or 1
        color = 1
    elif target.mode == "L" or target.mode == "P": #grayscale, 0 to 255
        color = 255
    elif target.mode == "RGB":
        color = (255, 255, 255)
    elif target.mode == "HSV":
        color = (0,0,255)
    elif target.mode == "CMYK":
        color = (0,0,0,0)
    elif target.mode == "RGBA":
        color = (255,255,255,255)
    elif target.mode == "LAB":
        color = (100, 0.01, -0.01)
    elif target.mode == "YCbCr":
        color = (1,0,0)
    else: # assuming signed integer
        color = 16777215

    # Draw bounding box
    output.rectangle(bb_coords, fill=color, outline=color)

    if args.image is not None:
        # Apply mask upon the -i image and save result
        input_img = Image.open(image_names[i])
        black = Image.new(input_img.mode, input_img.size) # Creating a new image with the default black background
        mask = target.convert("1") # transform bounding box image to black and white
        mask = mask.resize(input_img.size, Image.BICUBIC) #Adjust to input size
        black.paste(input_img, mask=mask)
        black.save(output_names[i])

    else:
        # Save output
        target.save(output_names[i])
