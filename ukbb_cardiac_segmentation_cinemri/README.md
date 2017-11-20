## Overview

Code for segmenting cardiovascular magnetic resonance (CMR) images from the [UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/) using fully convolutional networks.

**Note** This repository only contains the code, not the imaging data. To know more about how to access the UK Biobank imaging data, please go to the [UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/) website. Researchers can [apply](http://www.ukbiobank.ac.uk/register-apply/) to use the UK Biobank data resource for health-related research in the public interest.

## Usage

**A quick demo** You can run a quick demo:
```
python3 demo.py
```
There is one parameter in the script, *CUDA_VISIBLE_DEVICES*, which controls which GPU device to use on your machine. Currently, I set it to 0, which means the first GPU on your machine.

This script will download two exemplar short-axis cardiac MR images and a pre-trained network, then segment the left and right ventricles using the network, saving the segmentation results *seg_sa.nii.gz* and also saving the clinical measures in a spreadsheet *clinical_measure.csv*, including the left ventricular end-diastolic volume (LVEDV), end-systolic volume (LVESV), myocardial mass (LVM) and the right ventricular end-diastolic volume (RVEDV), end-systolic volume (RVESV). The script will also download exemplar long-axis cardiac MR images and segment the left and right atria.

**To know more** If you want to know more about how the network works and how it is trained, you can read these following files:
* network.py, which describes the neural network architecture;
* train_network.py, which trains a network on a dataset with both images and manual annotations;
* deploy_network.py, which deploys the trained network onto new images. If you are interested in deploying the pre-trained network to more UK Biobank cardiac image set, this is the file that you need to read.

## References

We would like to thank all the UK Biobank participants and staff who make the CMR imaging dataset possible and also people from Queen Mary's University London and Oxford University who performed the hard work of manual annotation. In case you find the toolbox or a certain part of it useful, please consider giving appropriate credit to it by citing one or some of the papers here, which respectively describes the segmentation method [1] and the manual annotation dataset [2]. Thanks.

[1] W. Bai, et al. Human-level CMR image analysis with deep fully convolutional networks. arXiv:1710.09289. [arxiv](https://arxiv.org/abs/1710.09289)

[2] S. Petersen, et al. Reference ranges for cardiac structure and function using cardiovascular magnetic resonance (CMR) in Caucasians from the UK Biobank population cohort. Journal of Cardiovascular Magnetic Resonance, 19:18, 2017. [doi](https://doi.org/10.1186/s12968-017-0327-9)