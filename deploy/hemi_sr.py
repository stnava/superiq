# this script assumes the image have been brain extracted
import os.path
from os import path

try:
    threads = os.environ['cpu_threads']
except KeyError:
    threads = "8"
# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import sys
import ants
import antspynet
import tensorflow as tf
import glob as glob
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string
import ia_batch_utils as batch

def dap( x ):
    bbt = ants.image_read( antspynet.get_antsxnet_data( "biobank" ) )
    bbt = antspynet.brain_extraction( bbt, "t1v0" ) * bbt
    qaff=ants.registration( bbt, x, "AffineFast" )
    dapper = antspynet.deep_atropos( qaff['warpedmovout'], do_preprocessing=False )
    dappertox = ants.apply_transforms(
        x,
        dapper['segmentation_image'],
        qaff['fwdtransforms'],
        interpolator='genericLabel',
        whichtoinvert=[True]
    )
    return(  dappertox )



def main(config):
    c = config
    # input brain extracted target image
    ifn_bucket = c.input_bucket
    ifn_key = c.input_value
    data = 'data/'
    ifn = batch.get_s3_object(ifn_bucket, ifn_key, data)
    img = ants.image_read(ifn)

    model_bucket = c.model_bucket
    model_key = c.model_key
    model_path = batch.get_s3_object(model_bucket, model_key, data)
    mdl = tf.keras.models.load_model( model_path ) # FIXME - parameterize this

    template_bucket = c.template_bucket
    templatefn = batch.get_s3_object(template_bucket, c.template_base, data)
    templateBF1L  =batch.get_s3_object(template_bucket,  c.templateBF1L, data)
    templateBF2L  =batch.get_s3_object(template_bucket,  c.templateBF2L, data)
    templateBF1R  =batch.get_s3_object(template_bucket,  c.templateBF1R, data)
    templateBF2R  =batch.get_s3_object(template_bucket,  c.templateBF2R, data)
    templateCIT =batch.get_s3_object(template_bucket,  c.templateCIT, data)
    templateHemi=batch.get_s3_object(template_bucket,  c.templateHemi, data)
    # inputs CIT168 and associated corticospinal tract labels
    # output approximate CST and supplementary motor cortex
    #model_file_name = "/Users/stnava/code/superiq/models/SEGSR_32_ANINN222_bigTV3.h5"
    #mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this
    #ifn = "/Users/stnava/data/data_old/PPMI/sr_examples/PPMI/3777/20160706/MRI_T1/I769284/direct_regseg/PPMI-3777-20160706-MRI_T1-I769284-direct_regseg-SR.nii.gz"
    #tdir = "/Users/stnava/data/BiogenSuperRes/CIT168_Reinf_Learn_v1/"
    #prefix = '/tmp/temp_' # FIXME this is the output


    #templatefn = tdir + "CIT168_T1w_700um_pad_adni0.nii.gz"
    #templateBF = glob.glob(tdir+"CIT168_basal_forebrain_adni_prob*gz")
    #templateBF.sort()
    #templateCIT = ants.image_read( tdir + "det_atlas_25_pad_LR_adni.nii.gz" )
    #templateHemi = ants.image_read( tdir + "CIT168_T1w_700um_pad_HemisphereLabel_adni.nii.gz" )
    #img = ants.image_read( ifn )
    template = ants.image_read( templatefn )

    tdap = dap( template )
    idap = dap( img )
    maskinds=[2,3,4,5]

    # if a registration to template is already computed ( which it should be ) then
    # just apply the resulting transformations otherwise compute a new reg
    # existing registrations look like prefixWarp.nii.gz, prefixGenericAffine.mat
    imgcerebrum = ants.mask_image(idap,idap,maskinds,binarize=True).iMath("GetLargestComponent")
    temcerebrum = ants.mask_image(tdap,tdap,maskinds,binarize=True).iMath("GetLargestComponent")

    regsegits=[200,200,200,50]

    if c.environment=='dev':
        regsegits=[200,200,200,10]

    if c.do_reg:
        regits = [600,600,600,200,50]
        lregits = [100, 100, 100, 55]
        verber=False
        if c.environment=='dev':
            regits=[600,600,0,0,0]
            lregits=[600,60,0,0,0]
            verber=True
        reg = ants.registration(
            template * temcerebrum,
            img * imgcerebrum,
            type_of_transform="SyN",
            grad_step = 0.20,
            syn_metric='CC',
            syn_sampling=2,
            reg_iterations=regits,
            outprefix=prefix,
            verbose=verber )

    # 1 is left, 2 is right
    hemiS = ants.apply_transforms(
        img,
        templateHemi,
        reg['invtransforms'],
        interpolator='genericLabel',
    )
    # these are the standard CIT labels
    citS = ants.apply_transforms(
        img,
        templateCIT,
        reg['invtransforms'],
        interpolator='genericLabel',
    )

    # these are the new BF labels
    bfprobs=[]
    bftot = img * 0.0
    templateBF = [templateBF1L, templateBF1R, templateBF2L,  templateBF2R]
    for x in templateBF:
        bfloc = ants.image_read(  x )
        bfloc = ants.apply_transforms( img, bfloc, reg['invtransforms'], interpolator='linear' )
        bftot = bftot + bfloc
        bfprobs.append( bfloc )

    # the segmentation is gained by thresholding each BF prob at 0.25 or thereabouts
    # and multiplying by imgcerebrum

    # write out at OR:
    prefix=c.output_file_prefix
    ants.image_write( img*imgcerebrum, prefix+'cerebrum.nii.gz' )
    ants.image_write( idap, prefix+'tissueSegmentation.nii.gz' )
    ants.image_write( hemiS, prefix+'hemisphere.nii.gz' )
    ants.image_write( citS, prefix+'CIT168Labels.nii.gz' )
    ants.image_write( bfprobs[0], prefix+'bfprob1left.nii.gz' )
    ants.image_write( bfprobs[1], prefix+'bfprob1right.nii.gz' )
    ants.image_write( bfprobs[2], prefix+'bfprob2left.nii.gz' )
    ants.image_write( bfprobs[3], prefix+'bfprob2right.nii.gz' )

    cortLR = ants.threshold_image( idap, 2, 2 ) * hemiS

    # get SNR of WM
    wmseg = ants.threshold_image( idap, 3, 3 )
    wmMean = img[ wmseg == 1 ].mean()
    wmStd = img[ wmseg == 1 ].std()
    # get SNR wrt CSF
    csfseg = ants.threshold_image( idap, 1, 1 )
    csfStd = img[ csfseg == 1 ].std()

    # this will give us super-resolution over the whole image and also SR cortex
    # however, it may be better to just re-run the seg on the output SR image
    # this took around 100GB RAM - could also do per lobe, less RAM but more time (probably)
    srseg = super_resolution_segmentation_per_label(
        img,
        cortLR,
        [2,2,2],
        sr_model=mdl,
        segmentation_numbers=[1,2],
        dilation_amount=2
    )
    idapSR = dap( srseg['super_resolution'] )
    wmsegSR = ants.threshold_image( idapSR, 3, 3 )
    wmMeanSR = srseg['super_resolution'][ wmsegSR == 1 ].mean()
    wmStdSR = srseg['super_resolution'][ wmsegSR == 1 ].std()
    csfsegSR = ants.threshold_image( idapSR, 1, 1 )
    csfStdSR = srseg['super_resolution'][ csfsegSR == 1 ].std()
    wmSNR = wmMean/wmStd
    wmcsfSNR = wmMean/csfStd
    wmSNRSR = wmMeanSR/wmStdSR
    wmcsfSNRSR = wmMeanSR/csfStdSR
    snrdf = {
        'WMSNROR': wmSNR,
        'WMCSFSNROR': wmcsfSNR,
        'WMSNRSR': wmSNRSR,
        'WMCSFSNRSR': wmcsfSNRSR,
    }
    df = pd.DataFrame(snrdf, index=[0])
    df.to_csv(prefix + 'wmsnr.csv', index=False)
    ants.image_write( srseg['super_resolution_segmentation'], prefix+'corticalSegSR.nii.gz' )
    ants.image_write( srseg['super_resolution'], prefix+'SR.nii.gz' )
    batch.handle_outputs(
        c.output_bucket,
        c.output_prefix,
        c.input_value,
        c.process_name,
    )

if __name__ == "__main__":
    config = sys.argv[1]
    config = batch.LoadConfig(config)
    main(config)
