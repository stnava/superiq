# superiq

## super-resolution image quantitation

make super-resolution fly: quantitatively and at scale

### quantitative methods for super-resolution images

* task-specific super-resolution
    * global tissue segmentation with priors
    * local cortical segmentation
    * hippocampus segmentation
    * basal forebrain segmentation
    * deep brain structure segmentation
        * substantia nigra
        * caudate, putamen, etc

* general purpose segmentation:
    * local joint label fusion for arbitrary segmentation libraries
    * local joint label fusion for single templates with augmentation

tests provide a good example of use cases.

* install: `python setup.py install`

* test: `python tests/test_segmentation.py`

## TODO

* documentation

* testing
    * figure out how to distribute sr models

* ....
