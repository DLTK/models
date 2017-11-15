import SimpleITK as sitk
import tensorflow as tf
import os

from dltk.io.augmentation import *
from dltk.io.preprocessing import *


def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.
    
    Args:
        file_references (list): A list of lists containing file references, such as [['id_0', 'image_filename_0', target_value_0], ..., ['id_N', 'image_filename_N', target_value_N]].
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or PREDICT.
        params (dict, optional): A dictionary to parameterise read_fn ouputs (e.g. reader_params = {'n_examples': 10, 'example_size': [64, 64, 64], 'extract_examples': True}, etc.).
    
    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader. 
    """
    
    def _augment(img, lbl):
        """An image augmentation function. 
        
        Args:
            img (np.array): Input image to be augmented. 
            lbl (np.array): Corresponding label to the input image. 
        
        Returns:
            np.array, np.array: The augmented image and corresponding label.
        """
        
        img = add_gaussian_offset(img, sigma=1.0)
        for a in range(3):
            [img, lbl] = flip([img, lbl], axis=a)
        
        return img, lbl
    
    
    def _map_labels(lbl, convert_to_protocol=False):
        """Map dataset specific label id protocols to consecutive integer ids for training and back.

            iFind segment ids:
                0 background
                2 brain
                9 placenta
                10 uterus ROI
            
        Args:
            lbl (np.array): A label map to be converted.
            convert_to_protocol (bool, optional) A flag to determine to convert from or to the protocol ids.

        Returns:
            np.array: The converted label map

        """

        ids = [0, 2]

        out_lbl = np.zeros_like(lbl)

        if convert_to_protocol:

            # Map from consecutive ints to protocol labels
            for i in range(len(ids)):
                out_lbl[lbl==i] = ids[i]
        else:

            # Map from protocol labels to consecutive ints
            for i in range(len(ids)):
                out_lbl[lbl==ids[i]] = i

        return out_lbl


    for f in file_references:

        # Read the image nii with sitk
        img_id = f[0]
        img_fn = f[1]
        img_sitk = sitk.ReadImage(str(img_fn))
        img = sitk.GetArrayFromImage(img_sitk)
        

        # Normalise volume image
        img = whitening(img)

        # Create a 4D image (i.e. [x, y, z, channels])
        images = np.expand_dims(img, axis=-1).astype(np.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}, 'labels': None}

        # Read the label nii with sitk
        lbl_fn = f[2]
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_fn))).astype(np.int32)
        
        # Map the label ids to consecutive integers
        lbl = _map_labels(lbl)
        
        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images, lbl = _augment(images, lbl)
        
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images, lbl = extract_class_balanced_example_array(images, lbl, example_size=params['example_size'],
                                                               n_examples=params['n_examples'], classes=2)
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)}, 'labels': {'y': lbl[e].astype(np.int32)}}
        else:
            yield {'features': {'x': images}, 'labels': {'y': lbl}, 'sitk': img_sitk, 'img_id': img_id}
    return