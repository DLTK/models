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
"""
    This script demonstrates the segmentation of a test cardiac MR image using
    a pre-trained neural network.
    """
import os, urllib.request


if __name__ == '__main__':
    # Set the GPU device id
    CUDA_VISIBLE_DEVICES = 0

    # Set the seq_name to 'sa', 'la_2ch' or 'la_4ch' for different imaging sequence
    for seq_name in ['sa', 'la_2ch', 'la_4ch']:
        print('Demo for {0} imaging sequence ...'.format(seq_name))

        # Download exemplar images
        URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'
        print('Downloading images ...')
        for i in [1, 2]:
            if not os.path.exists('demo_image/{0}'.format(i)):
                os.makedirs('demo_image/{0}'.format(i))
            f = 'demo_image/{0}/{1}.nii.gz'.format(i, seq_name)
            urllib.request.urlretrieve(URL + f, f)

        # Download the trained network
        print('Downloading the trained network ...')
        if not os.path.exists('trained_model'):
            os.makedirs('trained_model')
        for f in ['trained_model/FCN_{0}.meta'.format(seq_name),
                  'trained_model/FCN_{0}.index'.format(seq_name),
                  'trained_model/FCN_{0}.data-00000-of-00001'.format(seq_name)]:
            urllib.request.urlretrieve(URL + f, f)

        # Perform segmentation
        print('Performing segmentation ...')
        os.system('CUDA_VISIBLE_DEVICES={0} python3 deploy_network.py '
                  '--test_dir demo_image --dest_dir demo_image '
                  '--seq_name {1} --model_path trained_model/FCN_{1} '
                  '--process_seq --clinical_measures'.format(CUDA_VISIBLE_DEVICES, seq_name))
        print('Done.')