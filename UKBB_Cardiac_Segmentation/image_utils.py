# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage


def tf_categorical_accuracy(pred, truth):
    """ Accuracy metric """
    return tf.reduce_mean(tf.cast(tf.equal(pred, truth), dtype=tf.float32))


def tf_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = tf.cast(tf.equal(pred, k), dtype=tf.float32)
    B = tf.cast(tf.equal(truth, k), dtype=tf.float32)
    return 2 * tf.reduce_sum(tf.multiply(A, B)) / (tf.reduce_sum(A) + tf.reduce_sum(B))


def crop_image(image, cx, cy, size):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)), 'constant')
    elif crop.ndim == 4:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)), 'constant')
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def data_augmenter(image, label, shift, rotate, scale, intensity, flip):
    """
        Online data augmentation
        Perform affine transformation on image and label,
        which are 4D tensor of shape (N, H, W, C) and 3D tensor of shape (N, H, W).
    """
    image2 = np.zeros(image.shape, dtype=np.float32)
    label2 = np.zeros(label.shape, dtype=np.int32)
    for i in range(image.shape[0]):
        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [np.clip(np.random.normal(), -3, 3) * shift,
                     np.clip(np.random.normal(), -3, 3) * shift]
        rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
        scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply the affine transformation (rotation + scale + shift) to the image
        row, col = image.shape[1:3]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        for c in range(image.shape[3]):
            image2[i, :, :, c] = ndimage.interpolation.affine_transform(image[i, :, :, c],
                                                                        M[:, :2], M[:, 2], order=1)

        # Apply the affine transformation (rotation + scale + shift) to the label map
        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :],
                                                                 M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_val

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, ::-1, :, :]
                label2[i] = label2[i, ::-1, :]
            else:
                image2[i] = image2[i, :, ::-1, :]
                label2[i] = label2[i, :, ::-1]
    return image2, label2


def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def distance_metric(seg_A, seg_B, dx):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        """
    table_md = []
    table_hd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            _, contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            _, contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_md = np.mean(table_md) if table_md else None
    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_md, mean_hd
