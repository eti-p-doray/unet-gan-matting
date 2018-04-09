import os
import cv2
import random
import numpy


def combine_object_background(object_file, background_file, output_name):
    boder = 20

    foreground = cv2.imread(object_file, cv2.IMREAD_UNCHANGED)
    if foreground is None:
        return False
    alpha = cv2.split(foreground)[3]
    alpha = alpha.astype(float)
    alpha = cv2.copyMakeBorder(alpha,boder,boder,boder,boder,cv2.BORDER_CONSTANT,value=0)
    cv2.normalize(alpha, alpha, 0.0, 1.0, cv2.NORM_MINMAX)

    foreground = cv2.imread(object_file, cv2.IMREAD_COLOR)
    background = cv2.imread(background_file)
    if background is None:
        return False

    foreground = cv2.copyMakeBorder(foreground,boder,boder,boder,boder,cv2.BORDER_CONSTANT,value=[0,0,0])

    ratio = numpy.amax(numpy.divide(foreground.shape[0:2], background.shape[0:2]))
    background_size = numpy.ceil(numpy.multiply(background.shape[0:2], ratio)).astype(int)
    #print(numpy.multiply(background.shape[0:2], ratio).astype(int))
    background = cv2.resize(background, (background_size[1], background_size[0]))
    background = background[0:foreground.shape[0], 0:foreground.shape[1]]

    foreground = foreground.astype(float)
    background = background.astype(float)
    for i in range(0, 3):
        foreground[:,:,i] = numpy.multiply(alpha, foreground[:,:,i])
        background[:,:,i] = numpy.multiply(1.0 - alpha, background[:,:,i])
    outImage = numpy.add(foreground[:,:,0:3], background)

    cv2.imwrite(output_name, outImage)

    return True


def generate_trimap(object_file, trimap_name):
    boder = 20

    foreground = cv2.imread(object_file, cv2.IMREAD_UNCHANGED)
    if foreground is None:
        return False
    alpha = cv2.split(foreground)[3]
    alpha = alpha.astype(float)
    alpha = cv2.copyMakeBorder(alpha,boder,boder,boder,boder,cv2.BORDER_CONSTANT,value=0)
    cv2.normalize(alpha, alpha, 0.0, 1.0, cv2.NORM_MINMAX)

    _, inner_map = cv2.threshold(alpha, 0.999, 255, cv2.THRESH_BINARY)
    _, outer_map = cv2.threshold(alpha, 0.001, 255, cv2.THRESH_BINARY)

    inner_map = cv2.erode(inner_map, numpy.ones((5,5),numpy.uint8), iterations = 3)
    outer_map = cv2.dilate(outer_map, numpy.ones((5,5),numpy.uint8), iterations = 3)

    cv2.imwrite(trimap_name, inner_map + (outer_map - inner_map) /2)

def build_dataset(object_dir, background_dir, input_dir, trimap_dir):
    object_filenames = os.listdir(object_dir)
    background_filenames = os.listdir(background_dir)

    for i, object_file in enumerate(object_filenames):
        generate_trimap(
            os.path.join(object_dir, object_file),
            os.path.join(trimap_dir, str(i) + '_trimap.jpg'))

        backgrounds = random.sample(background_filenames, 20)
        for j, background_file in enumerate(backgrounds):
            print(i, j, object_file, background_file)
            combine_object_background(os.path.join(object_dir, object_file),
              os.path.join(background_dir, background_file),
              os.path.join(input_dir, str(i) + '_' + str(j) + '.jpg'))


if __name__ == "__main__":
    object_dir = os.path.join("data", "matting", "object")
    background_dir = os.path.join("data", "matting", "background")
    input_dir = os.path.join("data", "matting", "input")
    trimap_dir = os.path.join("data", "matting", "trimap")

    build_dataset(object_dir, background_dir, input_dir, trimap_dir)
