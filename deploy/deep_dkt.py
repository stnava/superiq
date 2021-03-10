import os
threads = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import tensorflow as tf

import antspynet
import math
import ants
import sys
from superiq.pipeline_utils import *
from superiq import deep_dkt, super_resolution_segmentation_per_label


def main(input_config):
    config = LoadConfig(input_config)
    if config.environment == 'prod':
        original_image_path = config.input_value

        input_image = get_pipeline_data(
                "brain_ext-bxtreg_n3.nii.gz",
                original_image_path,
                config.pipeline_bucket,
                config.pipeline_prefix,
                "/data"
        )

        input_image = ants.image_read(input_image)
    elif config.environment == 'val':
        input_path = get_s3_object(config.input_bucket, config.input_value, '/tmp')
        input_image = ants.image_read(input_path)

        image_base_name = input_path.split('/')[-1].split('.')[0]
        image_label_name = \
            config.atlas_label_prefix +  image_base_name + '_JLFSegOR.nii.gz'
        image_labels_path = get_s3_object(
            config.atlas_label_bucket,
            image_label_name,
            "/tmp",
        )
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

    wlab = config.wlab

    template = antspynet.get_antsxnet_data("biobank")
    template = ants.image_read(template)
    template =  template * antspynet.brain_extraction(template)

    sr_model = get_s3_object(config.model_bucket, config.model_key, "/tmp")
    mdl = tf.keras.models.load_model(sr_model)

    output_path = "/tmp/outputs/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_params={
        "target_image": input_image,
        "segmentation_numbers": wlab,
        "template": template,
        "sr_model": sr_model,
        "sr_params": config.sr_params,
        "output_path": output_path,
    }

    output = deep_dkt(**input_params)

    if config.environment == 'prod':
        handle_outputs(
            config.input_value,
            config.output_bucket,
            config.output_prefix,
            config.process_name,
            output_path,
        )

    elif config.environment == 'val':
        native_seg = output['nativeSeg']
        nativeGroundTruth = ants.image_read(image_labels_path)

        label_one_comp_set = config.label_one_labelset # [17, 53]
        label_one_gt = ants.threshold_image(
            nativeGroundTruth,
            label_one_comp_set[0],
            label_one_comp_set[0],
        )
        label_one = ants.threshold_image(
            native_seg,
            label_one_comp_set[1],
            label_one_comp_set[1],
        )
        dice_one = ants.label_overlap_measures(label_one_gt, label_one)
        dice_one = dice_one['MeanOverlap'][1]
        print(dice_one)

        label_two_comp_set = config.label_two_labelset # [1006, 2006]
        label_two_gt = ants.threshold_image(
            nativeGroundTruth,
            label_two_comp_set[0],
            label_two_comp_set[0],
        )
        label_two = ants.threshold_image(
            native_seg,
            label_two_comp_set[1],
            label_two_comp_set[1],
        )
        dice_two = ants.label_overlap_measures(label_two_gt, label_two)
        dice_two = dice_two['MeanOverlap'][1]
        print(dice_two)

        label_three_comp_set = config.label_three_labelset # [1006, 2006]
        label_three_gt = ants.threshold_image(
            nativeGroundTruth,
            label_three_comp_set[0],
            label_three_comp_set[0],
        )
        label_three = ants.threshold_image(
            native_seg,
            label_three_comp_set[1],
            label_three_comp_set[1],
        )
        dice_three = ants.label_overlap_measures(label_three_gt, label_three)
        dice_three = dice_three['MeanOverlap'][1]
        print(dice_three)

        brainName = []
        col1 = []
        col2 = []
        col3 = []

        brainName.append(target_image_base)
        col1.append(dice_one)
        col2.append(dice_two)
        col3.append(dice_three)

        dict = {
                'name': brainName,
                'hippocampus': col1,
                'entorhinal': col2,
                'parahippocampal': col3,
        }
        df = pd.DataFrame(dict)
        path = f"{image_base_name}_dice_scores.csv"
        df.to_csv("/tmp/" + path, index=False)
        s3 = boto3.client('s3')
        s3.upload_file(
            "/tmp/" + path,
            config.output_bucket,
            config.output_prefix + path,
        )
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
