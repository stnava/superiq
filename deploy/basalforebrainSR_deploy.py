import os
threads = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import ants
import sys
from superiq.pipeline_utils import *
from superiq import basalforebrainSR, basalforebrainOR, check_for_labels_in_image

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
    output_filename_sr_seg_init = output_filename  +  "_SR_seginit.nii.gz"
    output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
    output_filename_ortho_plot = output_filename  +  "_ortho_plot_sr.png"
    output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"
    
    bfSR = basalforebrainSR(
            input_image=input_image,
            template=template,
            templateL=templateL,
            model_path=model_path,
            atlas_images=brains,
            atlas_labels=brainsSeg,
            wlab=config.wlab,
            sr_params=config.sr_params,
            seg_params=config.seg_params,
    )
    sr = bfSR['SR_Img']  
    probseg = bfSR['SR_Seg'] 
   
    get_label_geo(
            probseg,
            sr, 
            config.process_name,
            config.input_value,
            resolution="SR",
    )
    
    plot_output(
        sr, 
        output_filename_ortho_plot, 
        probseg,
    )
    
    ants.image_write(sr, output_filename_sr)  
    ants.image_write(bfSR['SR_Seg_Init'], output_filename_sr_seg_init)  
    ants.image_write(probseg, output_filename_sr_seg)  

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
