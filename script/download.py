#!/usr/bin/python3
import argparse
import kaggle
import os
import zipfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='data', help='base datasets directory')
    args = parser.parse_args()

    base_datadir = args.dir

    if not os.path.isdir(base_datadir):
        os.mkdir(base_datadir)

    carvana_dir = os.path.join(base_datadir, 'carvana')
    tmp_dir = os.path.join(base_datadir, 'tmp')
    competition = 'carvana-image-masking-challenge'
    files = [
        'train.zip',
        'train_masks.zip'
    ]

    for filename in files:
        kaggle.api.competitionDownloadFile(competition, filename, path=tmp_dir)
        name, extension = os.path.splitext(filename)

        filepath = os.path.join(tmp_dir, filename)
        if extension == '.zip':
            with zipfile.ZipFile(filepath,"r") as zip_ref:
                zip_ref.extractall(carvana_dir)
            os.remove(filepath)
        else:
            os.rename(filepath, os.path.join(carvana_dir, filename))


if __name__ == "__main__":
    main()
