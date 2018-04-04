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
    whites = { #mode <=> value
        "1":1, #Binary image, 0 or 1
        "L":255, "P":255, #grayscale, 0 to 255
        "RGB":(255,255,255), "HSV":(0,0,255), "CMYK":(0,0,0,0),
        "RGBA":(255,255,255,255), "LAB":(100, 0.01, -0.01), "YCbCr":(1,0,0)
    }
    color = whites.get(target.mode, 16777215) #Default for signed integer

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
        black.close()
        input_img.close()

    else:
        # Save output
        target.save(output_names[i])
    target.close()
