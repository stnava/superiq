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
from superiq import basalforebrainSR
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
        print("Set up environment")
        shutil.rmtree("data", ignore_errors=True)
        os.makedirs("data")
        shutil.rmtree("outputs", ignore_errors=True)
        os.makedirs("outputs")

    def test_basalforebrainSR_config_isin(self):
        config = "configs/test_basalforebrain.json"
        basalforebrainSR(config=config)
        expected_output = "basalforebrainSR-SR_ortho_plot.png"
        outputs = os.listdir("outputs")
        testit = expected_output in outputs
        self.assertTrue(testit)

    def test_basalforebrainSR_local_isin(self):
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

        basalforebrainSR(
                templatefilename=template_image,
                templatesegfilename=template_labels,
                infn=input_n3,
                model_file_name=model,
                atlas_image_dir=brains[0:4],
                atlas_label_dir=brainsSeg[0:4],
                seg_params=seg_params
                )
        expected_output = "basalforebrainSR-SR_ortho_plot.png"
        outputs = os.listdir("outputs")
        testit = expected_output in outputs
        self.assertTrue(testit)

    def tearDown(self):
        shutil.rmtree("data")
        shutil.rmtree("outputs")



class TestModule_basalforebrainOR(unittest.TestCase):

    def setUp(self):
        print("Set up environment")
        shutil.rmtree("data", ignore_errors=True)
        os.makedirs("data")
        shutil.rmtree("outputs", ignore_errors=True)
        os.makedirs("outputs")

    def test_basalforebrainOR_config_isin(self):
        config = "configs/test_basalforebrain.json"
        basalforebrainOR(config=config)
        expected_output = "basalforebrain-OR_ortho_plot.png"
        outputs = os.listdir("outputs")
        testit = expected_output in outputs
        self.assertTrue(testit)

    def test_basalforebrainOR_local_isin(self):
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

        seg_params={
            "submask_dilation":5, "reg_iteration": [20,10,0],
            "searcher": 0, "radder": 2, "syn_sampling": 32, "syn_metric": "mattes",
            "max_lab_plus_one": True, "verbose": True}

        localbf = basalforebrain_segmentation(
            target_image=ants.image_read(input_n3),
            segmentation_numbers = wlab,
            template = ants.image_read(template_image),
            template_segmentation = ants.image_read(template_labels),
            library_intensity=images_to_list(brains[0:7]),
            library_segmentation=images_to_list(brainsSeg[0:7]),
            seg_params = seg_params
            )

        expected_segmentation = ants.image_read("expected_bf_segmentation.nii.gz")
        myol = ants.label_overlap_measures( expected_segmentation, localbf['probseg'] )
        overlap_test = False
        if myol['MeanOverlap'][0] > 0.95:
            overlap_test=True

        expected_output = "basalforebrain-OR_ortho_plot.png"
        outputs = os.listdir("outputs")
        testit = expected_output in outputs
        self.assertTrue(testit) and self.assertTrue(overlap_test)

    def tearDown(self):
        shutil.rmtree("data")
        shutil.rmtree("outputs")

if __name__ == "__main__":
    run_tests()
