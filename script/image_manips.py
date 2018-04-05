#!/usr/bin/python3
from PIL import Image, ImageDraw #NB : Requires Pillow : https://pillow.readthedocs.io/en/3.1.x/installation.html

def getBoundingBox(mask_image, padding=0):
    """Gets a bounding box around an masks's white content
    Should be given an opened PIL Image as |mask_image|.
    Can be given a |padding| around the bounding box. Should be positive for good looks
    """

    output = ImageDraw.Draw(mask_image)

    # Get bounding box (bb) around the non-zero pixels of the target mask
    bb = mask_image.getbbox() # returns (Left, Up, Right, Down)

    # Fit bounding box to image coordinates
    bb_coords = [0,0,0,0] # will contain 2 points
    bb_coords[0] = max(0, bb[0] - padding) # Point1 x
    bb_coords[1] = max(0, bb[1] - padding) # Point1 y
    bb_coords[2] = min(mask_image.size[0], bb[2] + padding) # Point2 x
    bb_coords[3] = min(mask_image.size[1], bb[3] + padding) # Point2 y

    # Set color to white
    whites = { #mode <=> value
        "1":1, #Binary image, 0 or 1
        "L":255, "P":255, #grayscale, 0 to 255
        "RGB":(255,255,255), "HSV":(0,0,255), "CMYK":(0,0,0,0),
        "RGBA":(255,255,255,255), "LAB":(100, 0.01, -0.01), "YCbCr":(1,0,0)
    }
    color = whites.get(mask_image.mode, 16777215) #Default for signed integer

    # Draw bounding box
    output.rectangle(bb_coords, fill=color, outline=color)

    return mask_image #With a drawn rectangle over it

def applyMask(input_img, mask_img):
    """Applies a mask over an image
    Both |input_img| and |mask_img| should be opened PIL Image
    """

    black = Image.new(input_img.mode, input_img.size) # Creating a new image with the default black background
    mask = mask_img.convert("1") # transform mask image to black and white
    mask = mask.resize(input_img.size, Image.BICUBIC) #Adjust to input size
    black.paste(input_img, mask=mask)
    return black

def resize(img, factor):
    """Resizes an image |img| by a certain |factor|.
    For instance a 400x200 img will be reduced to 200x100 if given factor 2.
    """
    newsize = (img.size[0] // factor, img.size[1] // factor)
    return img.resize(newsize)

def cropBlack(image, mask):
    """Removes black borders from an image, and adjust its mask
    """
    coords = image.getbbox()
    return image.crop(coords), mask.crop(coords)
