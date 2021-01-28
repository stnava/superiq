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
from superiq import basalforebrain
from superiq.pipeline_utils import *


def run_tests():
    unittest.main()

class TestModule_basalforebrain(unittest.TestCase):
    
    def setUp(self):
        print("Set up environment")
        shutil.rmtree("data", ignore_errors=True)
        os.makedirs("data")
        shutil.rmtree("outputs", ignore_errors=True)
        os.makedirs("outputs")

    def test_basalforebrain_config_isin(self):
        config = "configs/test_basalforebrain.json"
        basalforebrain(config=config)
        expected_output = "basalforebrain-SR_ortho_plot.png"
        outputs = os.listdir("outputs")  
        testit = expected_output in outputs
        self.assertTrue(testit)

    def test_basalforebrain_local_isin(self):
        input_n3 = get_s3_object(
            "invicro-data-shared", 
            "tests/ADNI_test/002_S_4473/20140227/T1w/000/ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz",       
            "data",
        )
        
        template_image = get_s3_object(
            "invicro-data-shared", 
            "adni_templates/adni_template.nii.gz", 
            "data",
        )
        
        template_labels = get_s3_object(
            "invicro-data-shared", 
            "adni_templates/adni_template_dkt_labels.nii.gz", 
            "data",
        )

        model = get_s3_object(
            "invicro-pipeline-inputs", 
            "models/SEGSR_32_ANINN222_3.h5", 
            "data",
        )
        
        atlas_bucket = "invicro-pipeline-inputs"
        atlas_image_prefx = "OASIS30/Brains/"
        atlas_label_prefix = "OASIS30/Segmentations/"
        atlas_image_keys = list_images(atlas_bucket, atlas_image_prefix)
        brains = [get_s3_object(atlas_bucket, k, "data") for k in atlas_image_keys]
        brains.sort()
        
        atlas_label_keys = list_images(atlas_bucket, atlas_label_prefix)
        brainsSeg = [get_s3_object(atlas_bucket, k, "atlas") for k in atlas_label_keys]
        brainsSeg.sort()
        
        basalforebrain(
                templatefilename=template_image,
                templatesegfilename=template_labels,
                infn=input_n3,
                model_file_name=model,
                atlas_image_dir=brains,
                atlas_label_dir=brainsSeg,
                )
        expected_output = "basalforebrain-SR_ortho_plot.png"
        outputs = os.listdir("outputs")  
        testit = expected_output in outputs
        self.assertTrue(testit)

    def tearDown(self):
        shutil.rmtree("data")
        shutil.rmtree("outputs")

if __name__ == "__main__":
    run_tests()
