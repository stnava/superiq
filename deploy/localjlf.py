import os
threads = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import tensorflow as tf

import math
import ants
import sys
from superiq import native_to_superres_ljlf_segmentation
from superiq import check_for_labels_in_image
from superiq import super_resolution_segmentation_per_label
import ia_batch_utils as batch

def local_jlf(input_config):
    config = batch.LoadConfig(input_config)
    tdir = "data"
    if config.environment == "prod":
        input_image = batch.get_s3_object(
            config.input_bucket,
            config.input_value,
            tdir,
        )

        atlas_image_keys = batch.list_objects(config.atlas_bucket, config.atlas_prefix)
        brains = [batch.get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_image_keys]
        brains.sort()
        brains = [ants.image_read(i) for i in brains]

        atlas_label_keys = batch.list_objects(config.atlas_bucket, config.atlas_labels)
        brainsSeg = [batch.get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_label_keys]
        brainsSeg.sort()
        brainsSeg = [ants.image_read(i) for i in brainsSeg]
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

    input_image = ants.image_read(input_image)

    # Noise Correction
    input_image = ants.denoise_image(input_image)
    input_image = ants.iMath(input_image, 'TruncateIntensity', 0.000001, 0.995)

    wlab = config.wlab
    template = batch.get_s3_object(config.template_bucket, config.template_key, "data")
    template = ants.image_read(template)

    templateL = batch.get_s3_object(config.template_bucket, config.template_labels, "data")
    templateL = ants.image_read(templateL)

    model_path = batch.get_s3_object(config.model_bucket, config.model_key, "models")
    mdl = tf.keras.models.load_model(model_path)

    havelabels = check_for_labels_in_image( wlab, templateL )

    if not havelabels:
        raise Exception("Label missing from the template")

    output_filename = config.output_folder
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    output_filename_sr = output_filename + "SR.nii.gz"
    output_filename_srOnNativeSeg = output_filename  +  "srOnNativeSeg.nii.gz"
    output_filename_sr_seg = output_filename  +  "SR_seg.nii.gz"
    output_filename_nr_seg = output_filename  +  "NR_seg.nii.gz"

    output_filename_ortho_plot_sr = output_filename  +  "ortho_plot_SR.png"
    output_filename_ortho_plot_srOnNativeSeg = output_filename  +  "ortho_plot_srOnNativeSeg.png"
    output_filename_ortho_plot_nrseg = output_filename  +  "ortho_plot_NRseg.png"
    output_filename_ortho_plot_srseg = output_filename  +  "ortho_plot_SRseg.png"

    output = native_to_superres_ljlf_segmentation(
            target_image = input_image, # n3 image
            segmentation_numbers = wlab,
            template = template,
            template_segmentation = templateL,
            library_intensity = brains,
            library_segmentation = brainsSeg,
            seg_params = config.seg_params,
            seg_params_sr = config.seg_params_sr,
            sr_params = config.sr_params,
            sr_model = mdl,
    )
    # SR Image
    sr = output['srOnNativeSeg']['super_resolution']
    ants.image_write(
            sr,
            output_filename_sr,
    )
    ants.plot_ortho(
            ants.crop_image(sr),
            flat=True,
            filename=output_filename_ortho_plot_sr,
    )

    # SR on native Seg
    srOnNativeSeg = output['srOnNativeSeg']['super_resolution_segmentation']
    ants.image_write(
            srOnNativeSeg,
            output_filename_srOnNativeSeg,
    )
    srnsdf = ants.label_geometry_measures(srOnNativeSeg)
    prepare_dynamo_outputs(
        srnsdf,
        config.input_value,
        config.batch_id,
        config.version,
        "super_resolution_on_native_seg",
        "SR"
    )

    cmask = ants.threshold_image(srOnNativeSeg, 1, math.inf).morphology('dilate', 4)
    ants.plot_ortho(
            ants.crop_image(sr, cmask),
            overlay=ants.crop_image(srOnNativeSeg, cmask),
            flat=True,
            filename=output_filename_ortho_plot_srOnNativeSeg,
    )

    # SR Seg
    srSeg = output['srSeg']['segmentation']
    ants.image_write(
            srSeg,
            output_filename_sr_seg,
    )

    srdf = ants.label_geometry_measures(srSeg)
    prepare_dynamo_outputs(
        srdf,
        config.input_value,
        config.batch_id,
        config.version,
        "super_resolution",
        "SR"
    )

    cmask = ants.threshold_image(srSeg, 1, math.inf).morphology('dilate', 4)
    ants.plot_ortho(
            ants.crop_image(sr, cmask),
            overlay=ants.crop_image(srSeg, cmask),
            flat=True,
            filename=output_filename_ortho_plot_srseg,
    )

    # Native Seg
    nativeSeg = output['nativeSeg']['segmentation']
    ants.image_write(
            nativeSeg,
            output_filename_nr_seg,
    )

    nrdf = ants.label_geometry_measures(nativeSeg)
    prepare_dynamo_outputs(
        nrdf,
        config.input_value,
        config.batch_id,
        config.version,
        "original_resolution",
        "OR"
    )

    cmask = ants.threshold_image(nativeSeg, 1, math.inf).morphology('dilate', 4)
    ants.plot_ortho(
            ants.crop_image(input_image, cmask),
            overlay=ants.crop_image(nativeSeg, cmask),
            flat=True,
            filename=output_filename_ortho_plot_nrseg,
    )
    if  config.environment == "prod":
        handle_outputs(
            config.input_value,
            config.output_bucket,
            config.output_prefix,
            config.process_name,
        )
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

def prepare_dynamo_outputs(lgmdf, input_value, batch_id, version, name, resolution):
    volumes = lgmdf[['Label', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']]

    volumes = volumes.to_dict('records')

    split = input_value.split('/')[-1].split('.')[0]
    rec = {}
    rec['originalimage'] = split
    rec['hashfields'] = ['originalimage', 'process', 'batchid', 'data']
    rec['batchid'] = batch_id
    rec['project'] = split[0]
    rec['subject'] = split[1]
    rec['date'] = split[2]
    rec['modality'] = split[3]
    rec['repeat'] = split[4]
    rec['process'] = 'local_jlf'
    rec['version'] = version
    rec['name'] = name
    rec['extension'] = ".nii.gz"
    rec['resolution'] = resolution
    for vol in volumes:
        vol.pop('Label', None)
        for k, v in vol.items():
            rec['data'] = {}
            rec['data']['label'] = 1
            rec['data']['key'] = k
            rec['data']['value'] = v
            batch.write_to_dynamo(rec)


if __name__ == "__main__":
    config = sys.argv[1]
    local_jlf(config)
