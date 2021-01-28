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

def basalforebrain_segmentation(
        target_image,
        template,
        template_segmentation,
        atlas_image_list,
        atlas_segmentation_list,
        seg_params={
            "wlab":[75,76], "submask_dilation":20, "reg_iteration": [100,50,10],
            "searcher": 2, "radder": 3, "syn_sampling": 2, "syn_metric": "CC",
            "max_lab_plus_one": False, "verbose": True},
        forward_transforms=None,
        output_filename=None,
        ):
    """
    Basal forebrain segmentation on a t1-weighted MR image of the brain.

    Arguments
    ---------

    target_image : ANTsImage
        input n3 image

    template : ANTsImage
        template image

    template_segmentation : ANTsImage
        template labels image

    atlas_image_list : list of ANTsImages
        the intensity images

    atlas_segmentation_list : list of ANTsImages
        the segmentation images

    seg_params : dict
        dict containing the variable parameters for the ljlf parcellation call.
        The parameters are:
            {"wlab":list, "submask_dilation":int, "reg_iteration": list,
            "searcher": int, "radder": int, "syn_sampling": int, "syn_metric": string,
            "max_lab_plus_one": bool, "verbose": bool}

    forward_transforms : dictionary
        output of ants.registration (optional). if not present, registration
        is computed within the function.

    output_filename : string
        passed to joint_label_fusion - stores its output

    Example
    -------
    >>> basalforebrain_segmentation(
            target_image=ants.image_read("data/input_n3_image.nii.gz"),
            template=ants.image_read("data/template_image.nii.gz"),
            template_segmentation=ants.image_read("data/template_label_image.nii.gz"),
            atlas_image_list=glob.glob("data/atlas_images/*"),
            atlas_segmentations=glob.glob"data/atlas_labels/*"),
            seg_params={
                "wlab":[75,76], "submask_dilation":20, "reg_iteration": [100,50,10],
                "searcher": 2, "radder": 3, "syn_sampling": 2, "syn_metric": "CC",
                "max_lab_plus_one": False, "verbose": True
            })
    """

    wlab = seg_params['wlab']

    # input data

    havelabels = check_for_labels_in_image( wlab, template_segmentation )

    if not havelabels:
        raise Exception("Label missing from the template")

    if forward_transforms is None:
        print("Registration")
        reg = ants.registration( target_image, template, 'SyN' )
        forward_transforms = reg['fwdtransforms']
        initlab0 = ants.apply_transforms( target_image, template_segmentation,
              forward_transforms, interpolator="genericLabel" )
    else:
        initlab0 = ants.apply_transforms( target_image, template_segmentation,
              forward_transforms, interpolator="genericLabel" )

    locseg = ljlf_parcellation(
            ants.iMath( target_image, "Normalize" ),
            segmentation_numbers=wlab,
            forward_transforms=forward_transforms,
            template=template,
            template_segmentationabels=template_segmentation,
            library_intensity = atlas_image_list,
            library_segmentation = atlas_segmentation_list,
            submask_dilation=seg_params['submask_dilation'],  # a parameter that should be explored
            searcher=seg_params['searcher'],  # double this for SR
            radder=seg_params['radder'],  # double this for SR
            reg_iterations=seg_params['reg_iterations'], # fast test
            syn_sampling=seg_params['syn_sampling'],
            syn_metric=seg_params['syn_metric'],
            max_lab_plus_one=seg_params['max_lab_plus_one'],
            output_prefix=output_filename,
            verbose=seg_params['verbose'],
        )
    probs = locseg['ljlf']['ljlf']['probabilityimages']
    probability_labels = locseg['ljlf']['ljlf']['segmentation_numbers']
    # find proper labels
    whichprob75 = probability_labels.index(wlab[0])
    whichprob76 = probability_labels.index(wlab[1])
    probsum = ants.resample_image_to_target(probs[whichprob75], target_image ) + ants.resample_image_to_target(probs[whichprob76], target_image )
    probseg = ants.threshold_image( probsum, 0.3, 1.0 )
    mygeo = ants.label_geometry_measures( probseg, probsum )
    return { "probsum":probsum, "probseg":probseg,"labelgeo":mygeo }
