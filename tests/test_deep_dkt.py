import superiq
from superiq.pipeline_utils import *
import unittest
import ants
import antspynet

class TestDeepDKT(unittest.TestCase):

    def setUp(self):
        self.original_image = get_s3_object(
            "invicro-data-shared",
            "ADNI/127_S_0112/20160205/T1w/001/ADNI-127_S_0112-20160205-T1w-001.nii.gz",
            "/tmp",
        )
        self.input_image = get_pipeline_data(
            'n3.nii.gz',
            self.original_image,
            "eisai-basalforebrainsuperres2",
            "superres-pipeline/",
            "/tmp"
        )
        template = antspynet.get_antsxnet_data("biobank")
        self.template = ants.image_read(template)
    def test_deep_dkt(self):
        input_params={
            "target_image": ants.image_read(self.input_image),
            "segmentation_numbers": [],
            "template": self.template,
            "library_intensity": "",
            "library_segmentation": "",
            "seg_params": {},
            "forward_transforms": None,
            "output_filename": None,
        }
        output = superiq.deep_dkt(**input_params)
        self.assertTrue(output)


if __name__ == "__main__":
    unittest.main()
