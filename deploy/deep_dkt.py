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

        nativeGroundTruth = ants.image_read(image_labels_path)
        nativeGroundTruth = ants.mask_image(
            nativeGroundTruth,
            nativeGroundTruth,
            level = wlab,
            binarize=False
        )
        gtSR = super_resolution_segmentation_per_label(
            imgIn = ants.iMath(input_image, "Normalize"),
            segmentation = nativeGroundTruth, # usually, an estimate from a template, not GT
            upFactor = config.sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = wlab,
            dilation_amount = config.sr_params['dilation_amount'],
            verbose = config.sr_params['verbose']
        )
        nativeGroundTruthSR = gtSR['super_resolution_segmentation']
        nativeGroundTruthBinSR = ants.mask_image(
            nativeGroundTruthSR,
            nativeGroundTruthSR,
            wlab,
            binarize=True
        )
        ######
        srseg = ants.threshold_image(output['superresSeg']['probsum'], 0.5, math.inf )
        nativeOverlapSloop = ants.label_overlap_measures(
            nativeGroundTruth,
            output['nativeSeg']['segmentation']
        )
        srOnNativeOverlapSloop = ants.label_overlap_measures(
            nativeGroundTruthSR,
            output['srOnNativeSeg']['super_resolution_segmentation']
        )
        srOverlapSloop = ants.label_overlap_measures(
            nativeGroundTruthSR,
            output['superresSeg']['segmentation']
        )
        srOverlap2 = ants.label_overlap_measures( nativeGroundTruthBinSR, srseg )

        brainName = []
        dicevalNativeSeg = []
        dicevalSRNativeSeg = []
        dicevalSRSeg = []

        brainName.append(target_image_base)
        dicevalNativeSeg.append(nativeOverlapSloop["MeanOverlap"][0])
        dicevalSRNativeSeg.append( srOnNativeOverlapSloop["MeanOverlap"][0])
        dicevalSRSeg.append( srOverlapSloop["MeanOverlap"][0])

        dict = {
                'name': brainName,
                'diceNativeSeg': dicevalNativeSeg,
                'diceSRNativeSeg': dicevalSRNativeSeg,
                'diceSRSeg': dicevalSRSeg
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
