import os
import os.path
from os import path
import glob as glob

import tensorflow
import ants
import sys
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import check_for_labels_in_image
from superiq.pipeline_utils import *
from superiq import list_to_string
from superiq import basalforebrain_segmentation

def basalforebrainSR(
        input_image,
        template,
        templateL,
        model_path,
        atlas_images,
        atlas_labels,
        forward_transforms=None,
        wlab=[75,76],
        sr_params={"upFactor": [2,2,2], "dilation_amount": 12, "verbose":True},
        seg_params={
            "submask_dilation":20, "reg_iterations": [100,50,10],
            "searcher": 2, "radder": 3, "syn_sampling": 2, "syn_metric": "CC",
            "max_lab_plus_one": False, "verbose": True}
        ):
    """
    # TODO

    Arguments
    ---------
    input_image : ANTsImage
        the input n3 image

    template : ANTsImage
        the template image

    templateL : ANTsImage
        the template labels image

    model_path : string
        path to the model file

    atlas_images : list (ANTsImages)
        the sorted list of atlas images

    atlas_labels : list (ANTsImages)
        the sorted list of atlas label images

    forward_transforms : list (str)
        a list with the paths to the warp image and transform matrix if using 
        preregistered input image

    wlab : list (int)
        list of labels of interest in the target image

    sr_params : dict
        dict containing the variable parameters for the super resolution call.
        The parameters are: { "upFactor" : list, "dilation_amount": int, "verbose" : bool}

    seg_params : dict
        dict containing the variable parameters for the ljlf parcellation call.
        The parameters are:
            {"submask_dilation":int, "reg_iterations": list,
            "searcher": int, "radder": int, "syn_sampling": int, "syn_metric": string,
            "max_lab_plus_one": bool, "verbose": bool}

    Example
    -------
    # TODO 
    """

    outputs = {}

    brains = atlas_images
    brainsSeg = atlas_labels
    output_filename = "outputs/basalforebrainSR"

    imgIn = input_image
    mdl = tf.keras.models.load_model( model_path )

    havelabels = check_for_labels_in_image( wlab, templateL )

    if not havelabels:
        raise Exception("Label missing from the template")

    output_filename_sr = output_filename + "_SR.nii.gz"
    output_filename_sr_seg_init = output_filename  +  "_SR_seginit.nii.gz"
    output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
    output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"

    if forward_transforms is None:
        print("Registration")
        reg = ants.registration( imgIn, template, 'SyN' )
        forward_transforms = reg['fwdtransforms']
        initlab0 = ants.apply_transforms( imgIn, templateL,
              forward_transforms, interpolator="genericLabel" )
    else:
        initlab0 = ants.apply_transforms( imgIn, templateL,
              forward_transforms, interpolator="genericLabel" )

    srseg = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = initlab0,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = wlab,
        dilation_amount = sr_params['dilation_amount'],
        verbose = sr_params['verbose']
    )

    srseg_tx = ants.apply_transforms( srseg['super_resolution'], templateL,
        forward_transforms, interpolator="genericLabel" )
    
    outputs['SR_Img'] = srseg['super_resolution']
    outputs['SR_Seg_Init'] = srseg_tx

    seg_input = ants.iMath(srseg['super_resolution'], "Normalize")
    localbf = basalforebrain_segmentation( 
            target_image=srseg['super_resolution'],
            segmentation_numbers=wlab,
            template=template,
            template_segmentation=templateL, 
            library_intensity=atlas_images,
            library_segmentation=atlas_labels,
            seg_params=seg_params,
    )
    
   
    probseg = ants.threshold_image(localbf['probsum'], 0.5, 2)
    outputs['SR_Seg'] = probseg

def basalforebrainOR(
        input_image,
        template,
        templateL,
        atlas_images,
        atlas_labels,
        forward_transforms=None,
        wlab=[75,76],
        seg_params={
            "submask_dilation":20, "reg_iterations": [100,50,10],
            "searcher": 2, "radder": 3, "syn_sampling": 2, "syn_metric": "CC",
            "max_lab_plus_one": False, "verbose": True},
        postsegSR=False,
        model_path=None,
        sr_params={"upFactor": [2,2,2], "dilation_amount": 12, "verbose":True},
        ):
    """
    # TODO

    Arguments
    ---------
    input_image : ANTsImage
        the input n3 image

    template : ANTsImage
        the template image

    templateL : ANTsImage
        the template labels image

    atlas_images : list (ANTsImages)
        the sorted list of atlas images

    atlas_labels : list (ANTsImages)
        the sorted list of atlas label images

    forward_transforms : list (str)
        a list with the paths to the warp image and transform matrix if using 
        preregistered input image

    wlab : list (int)
        list of labels of interest in the target image

    seg_params : dict
        dict containing the variable parameters for the ljlf parcellation call.
        The parameters are:
            {"submask_dilation":int, "reg_iterations": list,
            "searcher": int, "radder": int, "syn_sampling": int, "syn_metric": string,
            "max_lab_plus_one": bool, "verbose": bool}

    Example
    -------
    # TODO 
    """
    if ((postsegSR) and (model_path is None)):
        raise Exception("To do post segmentation SR, a model path is required")

    outputs = {}

    brains = atlas_images
    brainsSeg = atlas_labels
    output_filename = "outputs/basalforebrainOR"

    imgIn = input_image

    havelabels = check_for_labels_in_image( wlab, templateL )

    if not havelabels:
        raise Exception("Label missing from the template")
    
    # Dont need this
    #output_filename_or = output_filename + "_OR.nii.gz"
    #output_filename_or_seg_init = output_filename  +  "_OR_seginit.nii.gz"
    #output_filename_or_seg = output_filename  +  "_OR_seg.nii.gz"
    #output_filename_or_seg_sr = output_filename + "_postseg_sr.nii.gz" 
    #output_filename_or_seg_csv = output_filename  + "_OR_seg.csv"

    if forward_transforms is None:
        print("Registration")
        reg = ants.registration( imgIn, template, 'SyN' )
        forward_transforms = reg['fwdtransforms']
        initlab0 = ants.apply_transforms( imgIn, templateL,
              forward_transforms, interpolator="genericLabel" )
    else:
        initlab0 = ants.apply_transforms( imgIn, templateL,
              forward_transforms, interpolator="genericLabel" )
    
    outputs['OR_Seg_Init'] = initlab0
    #ants.image_write( init_tx , output_filename_or_seg_init )
    imgOR = ants.iMath(imgIn, "Normalize")

    localbf = basalforebrain_segmentation( 
            target_image=imgOR,
            segmentation_numbers=wlab,
            template=template,
            template_segmentation=templateL, 
            library_intensity=atlas_images,
            library_segmentation=atlas_labels,
            seg_params=seg_params,
    )
    
    # Don't think we need this anymore
    #locseg = ljlf_parcellation(
    #        srseg['super_resolution'],
    #        segmentation_numbers=wlab,
    #        forward_transforms=forward_transforms,
    #        template=template,
    #        templateLabels=templateL,
    #        library_intensity = brains,
    #        library_segmentation = brainsSeg,
    #        submask_dilation=seg_params['submask_dilation'],  # a parameter that should be explored
    #        searcher=seg_params['searcher'],  # double this for SR
    #        radder=seg_params['radder'],  # double this for SR
    #        reg_iterations=seg_params['reg_iterations'], # fast test
    #        syn_sampling=seg_params['syn_sampling'],
    #        syn_metric=seg_params['syn_metric'],
    #        max_lab_plus_one=seg_params['max_lab_plus_one'],
    #        output_prefix=output_filename,
    #        verbose=seg_params['verbose'],
    #    )
    #probs = locseg['ljlf']['ljlf']['probabilityimages']
    #probability_labels = locseg['ljlf']['ljlf']['segmentation_numbers']
    # find proper labels
    #whichprob75 = probability_labels.index(wlab[0])
    #whichprob76 = probability_labels.index(wlab[1])
    #probseg = ants.threshold_image(
    #  ants.resample_image_to_target(probs[whichprob75], srseg['super_resolution'] ) +
    #  ants.resample_image_to_target(probs[whichprob76], srseg['super_resolution'] ),
    #  0.3, 1.0 )
   
    probseg = ants.threshold_image(localbf['probsum'], 0.5, 2)
    outputs['OR_Seg'] = probseg
    #ants.image_write( probseg,  output_filename_or_seg )

    if postsegSR:
        mdl = tf.keras.models.load_model( model_path )
        srseg = super_resolution_segmentation_per_label(
            imgIn = imgOR,
            segmentation = probseg,
            upFactor = sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = wlab,
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose']
        )
        outputs['OR_PostSeg_SR'] = srseg['super_resolution_segmentation']
    
    return outputs
