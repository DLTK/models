import SimpleITK as sitk
import tensorflow as tf
import numpy as np

from dltk.io.augmentation import extract_class_balanced_example_array


def read_fn(file_references, mode, params=None):
    """Summary

    Args:
        file_references (TYPE): Description
        mode (TYPE): Description
        params (TYPE): Description

    Returns:
        TYPE: Description
    """
    for f in file_references:
        img_fn = str(f[0])

        img_name = img_fn.split('/')[-1].split('.')[0]

        # Use a SimpleITK reader to load the multi channel
        # nii images and labels for training
        img_sitk = sitk.ReadImage(img_fn)
        images = sitk.GetArrayFromImage(img_sitk)

        images = np.expand_dims(images, axis=3)

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}, 'labels': None,
                   'img_name': img_name, 'sitk': img_sitk}
        else:
            lbl_fn = str(f[1])
            lbl = sitk.GetArrayFromImage(
                sitk.ReadImage(lbl_fn)).astype(np.int32)

            # Augment if used in training mode
            if mode == tf.estimator.ModeKeys.TRAIN:
                pass

            # Check if the reader is supposed to return
            # training examples or full images
            if params['extract_examples']:
                n_examples = params['n_examples']
                example_size = params['example_size']

                images, lbl = extract_class_balanced_example_array(
                    images, lbl, example_size=example_size,
                    n_examples=n_examples, classes=14)

                for e in range(len(images)):
                    yield {'features': {'x': images[e].astype(np.float32)},
                           'labels': {'y': lbl[e].astype(np.int32)}}
            else:
                yield {'features': {'x': images}, 'labels': {'y': lbl},
                       'img_name': img_name, 'sitk': img_sitk}

    return
