# CNN-Background-Removal
Background Removal based on U-Net

# Download dataset

[kaggle](https://github.com/Kaggle/kaggle-api) api is required, along with API credentials.

Datasets can be downloaded with:

`python3 script/download.py`

Refer to [carvana competition](https://www.kaggle.com/c/carvana-image-masking-challenge) for information on dataset.

## Get simpler inputs

In order to reduce the complexity of the task, we use a small python3 script that finds a padded bounding box around the image's object. This can then be used by the neural network. The script `MakeBB.py` is responsible for it, with the following usage :

```
usage: MakeBB.py [-h] [-i IMAGE] [-p PADDING] [--directory] target output

Make a bounding box style mask out of a mask

positional arguments:
  target       Path (with filename and extension) of the mask from which to
               make a bounding box
  output       Path, name and extension of where to save the output image

optional arguments:
  -h, --help   show this help message and exit
  -i IMAGE     Asks for the resulting mask to be applied upon an image, given
               in here as a path with filename and extension.
  -p PADDING   How much padding to add around the minimal bounding box
  --directory  Indicate that the target path and the output path (and the -i
               image path) is towards a directory of targets and one of
               outputs instead (and for -i a directory of images associated to
               the mask by default sort order). For outputs, will use same
               name as input.
```
