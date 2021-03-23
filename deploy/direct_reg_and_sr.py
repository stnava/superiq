# this script assumes the image have been brain extracted
import os.path
from os import path

threads = "32"
# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
import antspynet
import tensorflow as tf
import os
import sys
import pandas as pd
import numpy as np

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import ljlf_parcellation_one_template
from superiq import list_to_string
from superiq.pipeline_utils import *



def main(input_config):
    c = LoadConfig(input_config)
    tdir = "data"
    #tfn = tdir + "CIT168_T1w_700um_pad.nii.gz"
    #tfnl = tdir + "det_atlas_25_pad.nii.gz"
    tfn = get_s3_object(c.template_bucket, c.template_key, tdir)
    tfnl = get_s3_object(c.template_bucket, c.template_label_key, tdir)

    #sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
    #infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
    infn = get_pipeline_data(
        c.brain_extraction_suffix,
        c.input_value,
        c.pipeline_bucket,
        c.pipeline_prefix,
        tdir,
    )

    model_file_name = get_s3_object(c.model_bucket, c.model_key, tdir)

    if not os.path.exists(c.output_folder):
        os.makedirs(c.output_folder)
    output_filename = c.output_folder + "/"

    # input data
    imgIn = ants.image_read( infn )
    imgIn = ants.denoise_image( imgIn, noise_model='Rician' )
    imgIn = ants.iMath( imgIn, "TruncateIntensity", 0.00001, 0.9995 ).iMath("Normalize")

    template = ants.image_read(tfn)
    templateL = ants.image_read(tfnl)

    if c.brain_age:
        t1_preprocessing = antspynet.preprocess_brain_image(
            imgIn,
            truncate_intensity=(0.00001, 0.9995),
            do_brain_extraction=False,
            template="croppedMni152",
            template_transform_type="AffineFast",
            do_bias_correction=False,
            do_denoising=False,
            antsxnet_cache_directory="/tmp/",
            verbose=True
        )

        bage = antspynet.brain_age(
            t1_preprocessing['preprocessed_image'],
            do_preprocessing=False
        )
        ages = []
        for i in bage['brain_age_per_slice']:
            ages.append(i[0])
        average = np.average(ages)
        std = np.std(ages)
        record = {
            'avg_Brain_Age': [average, ],
            'std_Brain_Age': [std, ]
        }
        brain_age_df = pd.DataFrame(record)
        brain_age_df.to_csv(output_filename + 'brain_age.csv')


    mdl = tf.keras.models.load_model( model_file_name )

    # expected output data
    output_filename_jac = output_filename + "jacobian.nii.gz"
    output_filename_seg = output_filename + "ORseg.nii.gz"
    output_filename_sr = output_filename + "SR.nii.gz"
    output_filename_sr_seg = output_filename  +  "SR_seg.nii.gz"
    output_filename_sr_segljlf = output_filename  +  "SR_segljlf.nii.gz"
    output_filename_sr_seg_csv = output_filename  + "SR_seg.csv"
    output_filename_warped = output_filename  + "warped.nii.gz"

    regits = (600,600,600,200,50)
    lregits = (100, 100,100, 55)
    verber=False
    reg = ants.registration(
        template,
        imgIn,
        type_of_transform="SyN",
        grad_step = 0.20,
        syn_metric='CC',
        syn_sampling=2,
        reg_iterations=regits,
        verbose=verber
    )

    ants.image_write( reg['warpedmovout'], output_filename_warped )
    myjacobian = ants.create_jacobian_determinant_image(
        template,
        reg['fwdtransforms'][0],
        True
    )
    ants.image_write( myjacobian, output_filename_jac )

    inv_transforms = reg['invtransforms']
    initlab0 = ants.apply_transforms(
        imgIn,
        templateL,
        inv_transforms,
        interpolator="nearestNeighbor"
    )
    ants.image_write( initlab0, output_filename_seg )

    g1 = ants.label_geometry_measures(initlab0,imgIn)
    g1.to_csv(output_filename + 'OR-lgm.csv')

    sr_params = c.sr_params
    mynums=c.wlab

    srseg = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = initlab0,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = mynums,
        dilation_amount = sr_params['dilation_amount'],
        max_lab_plus_one = True,
        verbose = sr_params['verbose']
    )

    ljlfseg = ljlf_parcellation_one_template(
	img = srseg['super_resolution'],
        segmentation_numbers = mynums,
        forward_transforms = inv_transforms,
        template = template,
        templateLabels = templateL,
        templateRepeats = 8,
        submask_dilation = 6,
        searcher=1,
        radder=2,
        reg_iterations=lregits,
        syn_sampling=2,
        syn_metric='CC',
        max_lab_plus_one=True,
        deformation_sd=2.0,
        intensity_sd=0.1,
        output_prefix=output_filename,
        verbose=False,
    )

    ants.image_write( srseg['super_resolution'], output_filename_sr )

    ants.image_write( srseg['super_resolution_segmentation'], output_filename_sr_seg )

    ants.image_write( ljlfseg['segmentation'], output_filename_sr_segljlf )

    localregsegtotal = srseg['super_resolution'] * 0.0

    labels=[]
    vols=[]
    areas=[]
    for mylab in mynums:
        localprefix = output_filename + "synlocal_label" + str( mylab ) + "_"
        print(localprefix)
        cmskt = ants.mask_image( templateL, templateL, mylab, binarize=True ).iMath( "MD", 8 )
        cimgt = ants.crop_image( template, cmskt ) \
            .resample_image( [0.5,0.5,0.5],use_voxels=False, interp_type=0 )
        cmsk = ants.mask_image(
            ljlfseg['segmentation'],
            ljlfseg['segmentation'],
            mylab,
            binarize=True,
        ).iMath( "MD", 12 )
        cimg = ants.crop_image( srseg['super_resolution'], cmsk )
        rig = ants.registration( ants.crop_image( cmskt ) , cmsk, "Rigid" )
        syn = ants.registration(
            ants.iMath(cimgt,"Normalize"),
            cimg, # srseg['super_resolution'],
            type_of_transform="SyNOnly",
            reg_iterations=[200,200,200,50],
            initial_transform=rig['fwdtransforms'][0],
            syn_metric='cc',
            syn_sampling=2,
            outprefix=localprefix,
            verbose=False,
        )
        if len( syn['fwdtransforms'] ) > 1 :
            jimg = ants.create_jacobian_determinant_image(
                cimgt,
                syn['fwdtransforms'][0],
                True,
                False,
            )
            ants.image_write( jimg, localprefix + "jacobian.nii.gz" )
            cmskt = ants.mask_image( templateL, templateL, mylab, binarize=True )
            localregseg = ants.apply_transforms(
                srseg['super_resolution'],
                cmskt,
                syn['invtransforms'],
                interpolator='genericLabel'
            )
            labels.append( mylab )
            vols.append(
                ants.label_geometry_measures( localregseg, localregseg )['VolumeInMillimeters'][1]
            )
            areas.append(
                ants.label_geometry_measures(
                    localregseg,
                    localregseg,
                )['SurfaceAreaInMillimetersSquared'][1]
            )
            localregseg = localregseg * mylab
            ants.image_write( syn['warpedmovout'], localprefix + "_localreg.nii.gz" )
            ants.image_write( localregseg, localprefix + "_localregseg.nii.gz" )
            localregsegtotal = localregseg + localregsegtotal

    seg_labels = {
        "Label": labels,
        "VolumeInMillimeters": vols,
        'SurfaceAreaInMillimetersSquared': areas,
    }
    seg_labels = pd.DataFrame(seg_labels)
    seg_labels.to_csv(output_filename + "SR-lgm.csv", index=False)

    #g2 = ants.label_geometry_measures(
    #    srseg['super_resolution_segmentation'],
    #    srseg['super_resolution']
    #)
    #g2.to_csv(output_filename + 'SR-lgm.csv', index=False)

    handle_outputs(
        c.input_value,
        c.output_bucket,
        c.output_prefix,
        c.process_name,
    )

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
