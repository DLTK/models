from __future__ import division

import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import argparse


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # resample images to 2mm spacing with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
        int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
        int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def normalise(itk_image):
    # normalise and clip images

    np_img = sitk.GetArrayFromImage(itk_image)
    np_img = np.clip(np_img, -1000., 800.).astype(np.float32)
    np_img = (np_img + 1000.) / 900. - 1.
    s_itk_image = sitk.GetImageFromArray(np_img)
    s_itk_image.CopyInformation(itk_image)
    return s_itk_image


def split_data(files, path, no_split=True, no_label=False):
    if no_split:
        # use this for test or so
        imgs = [os.path.join(path, 'img', 'img{}.nii.gz'.format(f))
                for f in files]

        if not no_label:
            lbls = [os.path.join(path, 'label', 'label{}.nii.gz'.format(f))
                    for f in files]

            pd.DataFrame(data={'imgs': imgs, 'lbls': lbls}).to_csv(
                'no_split.csv', index=False)
        else:
            pd.DataFrame(data={'imgs': imgs}).to_csv(
                'no_split.csv', index=False)
    else:
        # split train data into train and val
        rng = np.random.RandomState(42)
        ids = [f[3:7] for f in files]
        validation = rng.choice(ids, 7)
        train = [f for f in ids if f not in validation]

        train_imgs = [os.path.join(path, 'img', 'img{}.nii.gz'.format(f))
                      for f in train]
        if not no_label:
            train_lbls = [os.path.join(
                    path, 'label', 'label{}.nii.gz'.format(f)) for f in train]

            pd.DataFrame(data={'imgs': train_imgs, 'lbls': train_lbls}).to_csv(
                'train.csv', index=False)
        else:
            pd.DataFrame(data={'imgs': train_imgs}).to_csv(
                'train.csv', index=False)

        val_imgs = [os.path.join(path, 'img', 'img{}.nii.gz'.format(f))
                    for f in validation]
        if not no_label:
            val_lbls = [os.path.join(path, 'label', 'label{}.nii.gz'.format(f))
                        for f in validation]

            pd.DataFrame(data={'imgs': val_imgs, 'lbls': val_lbls}).to_csv(
                'val.csv', index=False)
        else:
            pd.DataFrame(data={'imgs': val_imgs}).to_csv(
                'val.csv', index=False)


def preprocess(args):
    files = os.listdir(os.path.join(args.data_path, 'img'))

    split_data(files, args.output_path, args.no_split, args.no_label)

    if not os.path.exists(os.path.join(args.data_path, 'img')):
        os.makedirs(os.path.join(args.data_path, 'img'))

    if not args.no_label:
        if not os.path.exists(os.path.join(args.data_path, 'label')):
            os.makedirs(os.path.join(args.data_path, 'label'))

    for f in tqdm(files):
        fid = f[3:7]
        f1 = os.path.join(args.data_path, 'img', f)

        nii_f1 = sitk.ReadImage(f1)
        res_nii_f1 = resample_img(nii_f1)
        scaled = normalise(res_nii_f1)
        sitk.WriteImage(scaled, os.path.join(args.output_path, 'img', f))

        if not args.no_label:
            l1 = os.path.join(
                args.data_path, 'label', 'label{}.nii.gz'.format(fid))
            nii_l1 = sitk.ReadImage(l1)
            res_nii_l1 = resample_img(nii_l1, is_label=True)
            sitk.WriteImage(res_nii_l1, os.path.join(
                    args.output_path, 'label', 'label{}.nii.gz'.format(fid)))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Example: Synapse CT example preprocessing script')

    parser.add_argument('--data_path', '-d')

    parser.add_argument('--output_path', '-p')

    parser.add_argument('--no_split', '-s', default=False, action='store_true')

    parser.add_argument('--no_label', '-n', default=False, action='store_true')

    args = parser.parse_args()

    # Call training
    preprocess(args)
