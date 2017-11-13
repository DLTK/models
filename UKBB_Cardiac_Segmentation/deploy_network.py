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
import os, time, math
import numpy as np, nibabel as nib, pandas as pd
import tensorflow as tf
from image_utils import *


""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
tf.app.flags._global_parser.add_argument('--seq_name', choices=['sa', 'la_2ch', 'la_4ch'],
                                         default='sa', help="Sequence name.")
tf.app.flags.DEFINE_string('test_dir', '/vol/biomedic2/wbai/tmp/github/test',
                           'Path to the test set directory, under which images are organised in '
                           'subdirectories for each subject.')
tf.app.flags.DEFINE_string('dest_dir', '/vol/biomedic2/wbai/tmp/github/output',
                           'Path to the destination directory, where the segmentations will be saved.')
tf.app.flags.DEFINE_string('model_path', '/vol/biomedic2/wbai/tmp/github/model/FCN_sa.ckpt-50000',
                           'Path to the saved trained model.')
tf.app.flags.DEFINE_boolean('process_seq', True, "Process a time sequence of images.")
tf.app.flags.DEFINE_boolean('save_seg', True, "Save segmentation.")
tf.app.flags.DEFINE_boolean('clinical_measure', True, "Calculate clinical measures.")


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(FLAGS.model_path))
        saver.restore(sess, '{0}'.format(FLAGS.model_path))

        print('Start evaluating on the test set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(FLAGS.test_dir))
        processed_list = []
        table = []
        table_time = []
        for data in data_list:
            print(data)
            data_dir = os.path.join(FLAGS.test_dir, data)

            if FLAGS.process_seq:
                # Process the temporal sequence
                image_name = '{0}/{1}.nii.gz'.format(data_dir, FLAGS.seq_name)

                if not os.path.exists(image_name):
                    print('  Directory {0} does not contain an image with file name {1}. '
                          'Skip.'.format(data_dir, os.path.basename(image_name)))
                    continue

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                image = nim.get_data()
                X, Y, Z, T = image.shape
                orig_image = image

                print('  Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity rescaling
                image = rescale_intensity(image, (1, 99))

                # Prediction (segmentation)
                pred = np.zeros(image.shape)

                # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                # in the network will result in the same image size at each resolution level.
                X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                for t in range(T):
                    # Transpose the shape to NXYC
                    image_fr = image[:, :, :, t]
                    image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                    image_fr = np.expand_dims(image_fr, axis=-1)

                    # Evaluate the network
                    prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                                feed_dict={'image:0': image_fr, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                    pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                    pred[:, :, :, t] = pred_fr

                seg_time = time.time() - start_seg_time
                print('  Segmentation time = {:3f}s'.format(seg_time))
                table_time += [seg_time]
                processed_list += [data]

                # ED frame defaults to be the first time frame.
                # Determine ES frame according to the minimum LV volume.
                k = {}
                k['ED'] = 0
                if FLAGS.seq_name == 'sa':
                    k['ES'] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))
                else:
                    k['ES'] = np.argmax(np.sum(pred == 1, axis=(0, 1, 2)))
                print('  ED frame = {:d}, ES frame = {:d}'.format(k['ED'], k['ES']))

                # Save the segmentation
                if FLAGS.save_seg:
                    print('  Saving segmentation ...')
                    dest_data_dir = os.path.join(FLAGS.dest_dir, data)
                    if not os.path.exists(dest_data_dir):
                        os.makedirs(dest_data_dir)

                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    nib.save(nim2, '{0}/seg_{1}.nii.gz'.format(dest_data_dir, FLAGS.seq_name))

                    for fr in ['ED', 'ES']:
                        nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                                 '{0}/{1}_{2}.nii.gz'.format(dest_data_dir, FLAGS.seq_name, fr))
                        nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine),
                                 '{0}/seg_{1}_{2}.nii.gz'.format(dest_data_dir, FLAGS.seq_name, fr))

                # Evaluate the clinical measures
                if FLAGS.seq_name == 'sa' and FLAGS.clinical_measure:
                    print('  Evaluating clinical measures ...')
                    measure = {}
                    dx, dy, dz = nim.header['pixdim'][1:4]
                    volume_per_voxel = dx * dy * dz * 1e-3
                    density = 1.05

                    for fr in ['ED', 'ES']:
                        measure[fr] = {}
                        measure[fr]['LVV'] = np.sum(pred[:, :, :, k[fr]] == 1) * volume_per_voxel
                        measure[fr]['LVM'] = np.sum(pred[:, :, :, k[fr]] == 2) * volume_per_voxel * density
                        measure[fr]['RVV'] = np.sum(pred[:, :, :, k[fr]] == 3) * volume_per_voxel

                    line = [measure['ED']['LVV'], measure['ES']['LVV'],
                            measure['ED']['LVM'],
                            measure['ED']['RVV'], measure['ES']['RVV']]
                    table += [line]
            else:
                # Process ED and ES time frames
                image_ED_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ED')
                image_ES_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ES')
                if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                    print('  Directory {0} does not contain an image with file name {1} or {2}. '
                          'Skip.'.format(data_dir, os.path.basename(image_ED_name),
                                          os.path.basename(image_ES_name)))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)

                    # Read the image
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(image_name)
                    image = nim.get_data()
                    X, Y = image.shape[:2]
                    if image.ndim == 2:
                        image = np.expand_dims(image, axis=2)

                    print('  Segmenting {} frame ...'.format(fr))
                    start_seg_time = time.time()

                    # Intensity rescaling
                    image = rescale_intensity(image, (1, 99))

                    # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                    # in the network will result in the same image size at each resolution level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYC
                    image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                    image = np.expand_dims(image, axis=-1)

                    # Evaluate the network
                    prob, pred = sess.run(['prob:0', 'pred:0'],
                                          feed_dict={'image:0': image, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    seg_time = time.time() - start_seg_time
                    print('  Segmentation time = {:3f}s'.format(seg_time))
                    table_time += [seg_time]
                    processed_list += [data]

                    # Save the segmentation
                    if FLAGS.save_seg:
                        print('  Saving segmentation ...')
                        dest_data_dir = os.path.join(FLAGS.dest_dir, data)
                        if not os.path.exists(dest_data_dir):
                            os.makedirs(dest_data_dir)

                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        nib.save(nim2, '{0}/seg_{1}_{2}.nii.gz'.format(dest_data_dir, FLAGS.seq_name, fr))

                    # Evaluate the clinical measures
                    if FLAGS.seq_name == 'sa' and FLAGS.clinical_measure:
                        print('  Evaluating clinical measures ...')
                        dx, dy, dz = nim.header['pixdim'][1:4]
                        volume_per_voxel = dx * dy * dz * 1e-3
                        density = 1.05

                        measure[fr] = {}
                        measure[fr]['LVV'] = np.sum(pred == 1) * volume_per_voxel
                        measure[fr]['LVM'] = np.sum(pred == 2) * volume_per_voxel * density
                        measure[fr]['RVV'] = np.sum(pred == 3) * volume_per_voxel

                if FLAGS.clinical_measure and FLAGS.seq_name == 'sa':
                    line = [measure['ED']['LVV'], measure['ES']['LVV'],
                            measure['ED']['LVM'],
                            measure['ED']['RVV'], measure['ES']['RVV']]
                    table += [line]

        # Save the spreadsheet for the clinical measures
        if FLAGS.seq_name == 'sa' and FLAGS.clinical_measure:
            column_names = ['LVEDV (mL)', 'LVESV (mL)', 'LVM (g)', 'RVEDV (mL)', 'RVESV (mL)']
            df = pd.DataFrame(table, index=processed_list, columns=column_names)
            csv_name = os.path.join(FLAGS.dest_dir, 'clinical_measure.csv')
            print('  Saving clinical measures at {0} ...'.format(csv_name))
            df.to_csv(csv_name)

        if FLAGS.process_seq:
            print('Average segmentation time = {:.3f}s per sequence'.format(np.mean(table_time)))
        else:
            print('Average segmentation time = {:.3f}s per frame'.format(np.mean(table_time)))
        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
            process_time, len(processed_list), process_time / len(processed_list)))
