## Abdomninal organ segmentation from 3D CT data

### Contact and referencing this work
If there are any issues please contact the corresponding author of this implementation. If you employ this model in your work, please refer to this citation of the [paper](https://arxiv.org/abs/1711.06853).
```
@article{pawlowski2017dltk,
  title={{DLTK: State of the Art Reference Implementations for Deep Learning on Medical Images}},
  author={Pawlowski, Nick and Ktena, Sofia Ira and Lee, Matthew CH and Kainz, Bernhard and Rueckert, Daniel and Glocker, Ben and Rajchl, Martin},
  journal={arXiv preprint arXiv:1711.06853},
  year={2017}
}
```

### Important Notes
- Batch normalisation was employed before each ReLu non-linearity

### Data
The data can be downloaded after registration from the [challenge website](http://synapse.org/#!Synapse:syn3193805/wiki/217785).

Images and segmentations are read from a csv file in the format below. The original files (*.csv) is provided in this repo. 

These are parsed and extract tf.Tensor examples for training and evaluation in `reader.py` using a [SimpleITK](http://www.simpleitk.org/) for i/o of the .nii files.


### Usage
You can use the code (train.py) to train the model on the data yourself. Alternatively, we provide pretrained models here:
- [original submission](https://www.doc.ic.ac.uk/~np716/dltk_models/ct_synapse/orig_unet.tar.gz)
- [DLTK 0.2 asymetric U-Net](https://www.doc.ic.ac.uk/~np716/dltk_models/ct_synapse/asym_unet_balce_mom.tar.gz)
- [DLTK 0.2 FCN](https://www.doc.ic.ac.uk/~np716/dltk_models/ct_synapse/fcn_balce.tar.gz)

