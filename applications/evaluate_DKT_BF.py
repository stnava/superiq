import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import os.path
from os import path
import glob as glob
# set number of threads - this should be optimized per compute instance


import tensorflow
import ants
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import check_for_labels_in_image
from superiq import sort_library_by_similarity
from superiq import basalforebrain_segmentation

def images_to_list( x ):
    outlist = []
    for k in range(len(x)):
        outlist.append( ants.image_read( x[k] ) )
    return outlist

# user definitions here
tdir = "/Users/stnava/code/super_resolution_pipelines/data/OASIS30/"
brains = glob.glob(tdir+"Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"Segmentations/*")
brainsSeg.sort()
templatefilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template.nii.gz"
templatesegfilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template_dkt_labels.nii.gz"

overlaps = []
for k in range( len( brains ) ):
     brainsLocal=brains.copy()
     brainsSegLocal=brainsSeg.copy()
     del brainsLocal[k:(k+1)]
     del brainsSegLocal[k:(k+1)]
     localbf = basalforebrain_segmentation(
       target_image=ants.image_read(brains[k]),
       template = ants.image_read(templatefilename),
       template_segmentation = ants.image_read(templatesegfilename),
       atlas_image_list=images_to_list(brainsLocal),
       atlas_segmentation_list=images_to_list(brainsSegLocal),
       )
     overlaps.append( myol )
     # get overlap values
