# Deep Learning Models for Connectomics with EM images
----------------------------

The `fiber` branch of this fork of the original [EM-Network](https://github.com/donglaiw/EM-network/.git) repository is used to generate affinities parallel fiber segmentation. It has the following features over the original:

* AffinityDatasets can now ignore the '0' segment during training
* The training scripts `train_*` now save affinity images to the TensorBoard summary during training
* UnetFiber: a new network design