import sys
import os
import shutil
import argparse
import unittest
import warnings
import contextlib
import numpy as np
import numpy.testing as nptest

import unittest
import ants
from superiq import basalforebrainSR, basalforebrainOR
from superiq import basalforebrain_segmentation
from superiq.pipeline_utils import *


def run_tests():
    unittest.main()

def images_to_list( x ):
    outlist = []
    for k in range(len(x)):
        outlist.append( ants.image_read( x[k] ) )
    return outlist

class TestModule_basalforebrainSR(unittest.TestCase):

    def setUp(self):
        shutil.rmtree("data", ignore_errors=True)
        os.makedirs("data")

    def test_basalforebrainSR_local_isin(self):
        input_n3 = get_s3_object(
            "invicro-data-shared",
            "tests/ADNI_test/002_S_4473/20140227/T1w/000/ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz",
            "data",
        )

        template_image = get_s3_object(
            "invicro-pipeline-inputs",
            "adni_templates/adni_template.nii.gz",
            "data",
        )

        template_labels = get_s3_object(
            "invicro-pipeline-inputs",
            "adni_templates/adni_template_dkt_labels.nii.gz",
            "data",
        )

        model = get_s3_object(
            "invicro-pipeline-inputs",
            "models/SEGSR_32_ANINN222_3.h5",
            "data",
        )

        atlas_bucket = "invicro-pipeline-inputs"
        atlas_image_prefix = "OASIS30/Brains/"
        atlas_label_prefix = "OASIS30/Segmentations/"
        atlas_image_keys = list_images(atlas_bucket, atlas_image_prefix)
        brains = [get_s3_object(atlas_bucket, k, "data") for k in atlas_image_keys]
        brains.sort()
        atlas_images = [ants.image_read(i) for i in brains]

        atlas_label_keys = list_images(atlas_bucket, atlas_label_prefix)
        brainsSeg = [get_s3_object(atlas_bucket, k, "data") for k in atlas_label_keys]
        brainsSeg.sort()
        atlas_labels = [ants.image_read(i) for i in brainsSeg]

        outputs = basalforebrainSR(
                input_image=ants.image_read(input_n3),
                template=ants.image_read(template_image),
                templateL=ants.image_read(template_labels),
                model_path=model,
                atlas_images=atlas_images[0:4],
                atlas_labels=atlas_labels[0:4],
                wlab=[75,76]
                )
        
        self.assertTrue(outputs)

    def tearDown(self):
        shutil.rmtree("data")



class TestModule_basalforebrainOR(unittest.TestCase):

    def setUp(self):
        shutil.rmtree("data", ignore_errors=True)
        os.makedirs("data")

    def test_basalforebrainOR_local_isin(self):
        input_n3 = get_s3_object(
            "invicro-data-shared",
            "tests/ADNI_test/002_S_4473/20140227/T1w/000/ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz",
            "data",
        )

        template_image = get_s3_object(
            "invicro-pipeline-inputs",
            "adni_templates/adni_template.nii.gz",
            "data",
        )

        template_labels = get_s3_object(
            "invicro-pipeline-inputs",
            "adni_templates/adni_template_dkt_labels.nii.gz",
            "data",
        )

        model = get_s3_object(
            "invicro-pipeline-inputs",
            "models/SEGSR_32_ANINN222_3.h5",
            "data",
        )

        atlas_bucket = "invicro-pipeline-inputs"
        atlas_image_prefix = "OASIS30/Brains/"
        atlas_label_prefix = "OASIS30/Segmentations/"
        atlas_image_keys = list_images(atlas_bucket, atlas_image_prefix)
        brains = [get_s3_object(atlas_bucket, k, "data") for k in atlas_image_keys]
        brains.sort()
        atlas_images = [ants.image_read(i) for i in brains]

        atlas_label_keys = list_images(atlas_bucket, atlas_label_prefix)
        brainsSeg = [get_s3_object(atlas_bucket, k, "data") for k in atlas_label_keys]
        brainsSeg.sort()
        atlas_labels = [ants.image_read(i) for i in brainsSeg]

        outputs = basalforebrainOR(
                input_image=ants.image_read(input_n3),
                template=ants.image_read(template_image),
                templateL=ants.image_read(template_labels),
                atlas_images=atlas_images[0:4],
                atlas_labels=atlas_labels[0:4],
                wlab=[75,76]
                )
        
        self.assertTrue(outputs)
    
    def test_basalforebrainOR_postSegSR(self):
        input_n3 = get_s3_object(
            "invicro-data-shared",
            "tests/ADNI_test/002_S_4473/20140227/T1w/000/ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz",
            "data",
        )

        template_image = get_s3_object(
            "invicro-pipeline-inputs",
            "adni_templates/adni_template.nii.gz",
            "data",
        )

        template_labels = get_s3_object(
            "invicro-pipeline-inputs",
            "adni_templates/adni_template_dkt_labels.nii.gz",
            "data",
        )

        model = get_s3_object(
            "invicro-pipeline-inputs",
            "models/SEGSR_32_ANINN222_3.h5",
            "data",
        )

        atlas_bucket = "invicro-pipeline-inputs"
        atlas_image_prefix = "OASIS30/Brains/"
        atlas_label_prefix = "OASIS30/Segmentations/"
        atlas_image_keys = list_images(atlas_bucket, atlas_image_prefix)
        brains = [get_s3_object(atlas_bucket, k, "data") for k in atlas_image_keys]
        brains.sort()
        atlas_images = [ants.image_read(i) for i in brains]

        atlas_label_keys = list_images(atlas_bucket, atlas_label_prefix)
        brainsSeg = [get_s3_object(atlas_bucket, k, "data") for k in atlas_label_keys]
        brainsSeg.sort()
        atlas_labels = [ants.image_read(i) for i in brainsSeg]

        outputs = basalforebrainOR(
                input_image=ants.image_read(input_n3),
                template=ants.image_read(template_image),
                templateL=ants.image_read(template_labels),
                model_path=model,
                atlas_images=atlas_images[0:4],
                atlas_labels=atlas_labels[0:4],
                wlab=[75,76]
                )
        
        self.assertTrue(outputs)

    def tearDown(self):
        shutil.rmtree("data")

if __name__ == "__main__":
    run_tests()
