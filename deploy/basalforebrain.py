import ants
from superiq import basalforebrainSR
from superiq.pipeline_utils import *
import pandas as pd
import os

def main(input_config):
    config = LoadConfig(input_config)
    input_n3 = get_pipeline_data(
            "bxtreg_n3.nii.gz",
            config.input_value,
            config.pipeline_bucket,
            config.pipeline_prefix,
    )
    template_image = get_s3_data(

if __name__ == "__main__":
    pass
