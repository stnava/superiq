import os
threads = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import ants
import sys
from superiq.pipeline_utils import *
from superiq import native_to_superres_ljlf_segmentation, check_for_labels_in_image

def main(input_config):
    config = LoadConfig(input_config)
    input_image = get_pipeline_data(
            "brain_ext-bxtreg_n3.nii.gz",
            config.input_value,
            config.pipeline_bucket,
            config.pipeline_prefix,
    )
    input_image = ants.image_read(input_image) 
    wlab = config.wlab
    template = get_s3_object(config.template_bucket, config.template_key, "data")
    template = ants.image_read(template) 
    
    templateL = get_s3_object(config.template_bucket, config.template_label_key, "data")
    templateL = ants.image_read(templateL) 

    model_path = get_s3_object(config.model_bucket, config.model_key, "models")
    mdl = tf.keras.models.load_model(model_path) 


    atlas_image_keys = list_images(config.atlas_bucket, config.atlas_image_prefix)
    brains = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_image_keys]
    brains.sort()
    brains = [ants.image_read(i) for i in brains] 
    
    atlas_label_keys = list_images(config.atlas_bucket, config.atlas_label_prefix)
    brainsSeg = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_label_keys]
    brainsSeg.sort()
    brainsSeg = [ants.image_read(i) for i in brainsSeg] 
     
    havelabels = check_for_labels_in_image( wlab, templateL )
    
    if not havelabels:
        raise Exception("Label missing from the template")
    
    output_filename = "outputs/" + config.output_file_prefix 
    
    output_filename_sr = output_filename + "_SR.nii.gz"
    output_filename_srOnNativeSeg = output_filename  +  "_srOnNativeSeg.nii.gz"
    output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
    output_filename_nr_seg = output_filename  +  "_NR_seg.nii.gz"
    
    output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"
    output_filename_srOnNativeSeg_csv = output_filename  + "_srOnNativeSeg.csv"
    output_filename_nr_seg_csv = output_filename  + "_NR_seg.csv"
    output_filename_ortho_plot_sr = output_filename  +  "_ortho_plot_SR.png"
    output_filename_ortho_plot_srOnNativeSeg = output_filename  +  "_ortho_plot_srOnNativeSeg.png"
    output_filename_ortho_plot_nr = output_filename  +  "_ortho_plot_NR.png"
    
    output = native_to_superres_ljlf_segmentation(
            target_image = input_image, # n3 image
            segmentation_numbers = wlab,
            template = template,
            template_segmentation = templateL,
            library_intensity = brains,
            library_segmentation = brainsSeg,
            seg_params = config.seg_params,
            sr_params = config.sr_params,
            sr_model = mdl,
    )
  
    sr = output['srOnNativeSeg']['super_resolution'], 
    ants.image_write(
            output['srOnNativeSeg']['super_resolution'], 
            output_filename_sr,
    )
    
    ants.image_write(
            output['srOnNativeSeg']['super_resolution_segmentation'], 
            output_filename_srOnNativeSeg,
    )
    SRNSdf = ants.label_geometry_measures(output['srOnNativeSeg']['super_resolution_segmentation'])
    SRNSdf.to_csv(output_filename_srOnNativeSeg_csv, index=False) 
    plot_output(
        sr,
        output_filename_ortho_plot_srOnNativeSeg,
        output['srOnNativeSeg']['super_resolution_segmentation'], 
    )

    ants.image_write(
            output['srSeg'],
            output_filename_sr_seg,
    )
    SRdf = ants.label_geometry_measures(output['srSeg'])
    SRdf.to_csv(output_filename_sr_seg_csv, index=False) 
    plot_output(
        sr,
        output_filename_ortho_plot_sr,
        output['srSeg'], 
    )
    
    ants.image_write(
            output['nativeSeg'],
            output_filename_nr_seg,
    )
    NRdf = ants.label_geometry_measures(output['nativeSeg'])
    NRdf.to_csv(output_filename_nr_seg_csv, index=False) 
    plot_output(
        input_image,
        output_filename_ortho_plot_nr,
        output['nativeSeg'],
    )
    
    handle_outputs(
        config.input_value,
        config.output_bucket,
        config.output_prefix,
        config.process_name,
        env=config.environment,
    )

if __name__ == "__main__":
    config = sys.argv[1] 
    main(config)
