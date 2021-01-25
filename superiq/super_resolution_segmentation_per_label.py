
import numpy as np
import ants
import tensorflow as tf
import antspynet
import tempfile
import warnings


def check_for_labels_in_image( label_list, img ):
    imglabels = img.unique()
    isin = True
    for x in range( len( label_list ) ):
        isin = isin & ( label_list[x] in imglabels )
    return isin


def super_resolution_segmentation_per_label(
    imgIn,
    segmentation, # segmentation label image
    upFactor,
    sr_model,
    segmentation_numbers,
    dilation_amount = 6,
    probability_images=None, # probability list
    probability_labels=None, # the segmentation ids for the probability image
    verbose = False,
):
    """
    Apply a two-channel super resolution model to an image and probability pair.

    Arguments
    ---------
    imgIn : ANTsImage
        image to be upsampled

    segmentation : ANTsImage
        segmentation probability, n-ary or binary image

    upFactor : list
        the upsampling factors associated with the super-resolution model

    sr_model : tensorflow model
        for computing super-resolution

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    dilation_amount : integer
        amount to pad around each segmentation defining region over which to
        computer the super-resolution in each label

    probability_images : list of ANTsImages
        list of probability images

    probability_labels : integer list
        providing the integer values associated with each probability image

    verbose : boolean
        whether to show status updates

    Returns
    -------

    dictionary w/ following key/value pairs:
        `super_resolution` : ANTsImage
            super_resolution image

        `super_resolution_segmentation` : ANTsImage
            super_resolution_segmentation image

        `segmentation_geometry` : list of data frame types
            segmentation geometry for each label

        `probability_images` : list of ANTsImage
            segmentation probability maps


    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    >>> FIXME
    """
    newspc = ( np.asarray( ants.get_spacing( imgIn ) ) ).tolist()
    for k in range(len(newspc)):
        newspc[k] = newspc[k]/upFactor[k]
    imgup = ants.resample_image( imgIn, newspc, use_voxels=False, interp_type=0 )
    imgsrfull = imgup * 0.0
    weightedavg = imgup * 0.0
    problist = []
    seggeom = []
    for locallab in segmentation_numbers:
        if verbose:
            print( "SR-per-label:" + str( locallab ) )
        binseg = ants.threshold_image( segmentation, locallab, locallab )
        if ( binseg == 1 ).sum() == 0 :
            warnings.warn( "SR-per-label:" + str( locallab ) + 'does not exist' )
        # FIXME replace binseg with probimg and use minprob to threshold it after SR
        minprob="NA"
        if probability_images is not None:
            whichprob = probability_labels.index(locallab)
            probimg = probability_images[whichprob].resample_image_to_target( binseg )
            minprob = min( probimg[ binseg >= 0.5 ] )
        if verbose:
            print( "SR-per-label:" + str( locallab ) + " min-prob: " + str(minprob)  )
        binsegdil = ants.iMath( ants.threshold_image( segmentation, locallab, locallab ), "MD", dilation_amount )
        binsegdil2input = ants.resample_image_to_target( binsegdil, imgIn, interp_type='nearestNeighbor'  )
        imgc = ants.crop_image( imgIn, binsegdil2input ).iMath("Normalize")
        imgc = imgc * 255 - 127.5 # for SR
        imgch = ants.crop_image( binseg, binsegdil )
        imgch = ants.iMath( imgch, "Normalize" ) * 255 - 127.5 # for SR
        myarr = np.stack( [imgc.numpy(),imgch.numpy()],axis=3 )
        newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
        myarr = myarr.reshape( newshape )
        pred = sr_model.predict( myarr )
        imgsr = ants.from_numpy( tf.squeeze( pred[0] ).numpy())
        imgsr = ants.copy_image_info( imgc, imgsr )
        newspc = ( np.asarray( ants.get_spacing( imgsr ) ) * 0.5 ).tolist()
        ants.set_spacing( imgsr,  newspc )
        imgsrh = ants.from_numpy( tf.squeeze( pred[1] ).numpy())
        imgsrh = ants.copy_image_info( imgc, imgsrh )
        ants.set_spacing( imgsrh,  newspc )
        if locallab == segmentation_numbers[0]:
            imgsegjoin = imgup * 0.0
        # NOTE: this only works because we use sigmoid activation with binary labels
        # NOTE: we could also compute the minimum probability in the label and run
        # SR on the probability images
        imgsrhb = ants.threshold_image( imgsrh, 0.5, 1.0 ).iMath("GetLargestComponent")
        problist.append( imgsrh )
        temp = ants.resample_image_to_target( imgsrhb * locallab, imgup, interp_type='nearestNeighbor' )
        temp[ (temp > 0) * (imgsegjoin > 0) ] = 0 # FIXME zeroes out uncertain boundaries
        selector = (imgsegjoin == 0) * (temp > 0) # FIXME - this introduces some dependencies on ordering
        imgsegjoin[ selector ] = imgsegjoin[ selector ] + temp[ selector ]
        imgsr = antspynet.regression_match_image( imgsr, ants.resample_image_to_target(imgup,imgsr) )
        contribtoavg = ants.resample_image_to_target( imgsr*0+1, imgup, interp_type='nearestNeighbor' )
        weightedavg = weightedavg + contribtoavg
        imgsrfull = imgsrfull + ants.resample_image_to_target( imgsr, imgup, interp_type='nearestNeighbor' )
        seggeom.append( ants.label_geometry_measures( imgsrhb ))

    imgsrfull2 = imgsrfull
    selector = imgsrfull == 0
    imgsrfull2[ selector  ] = imgup[ selector ]
    weightedavg[ weightedavg == 0.0 ] = 1.0
    imgsrfull2=imgsrfull2/weightedavg
    imgsrfull2[ imgup == 0 ] = 0
    return {
        "super_resolution": imgsrfull2,
        "super_resolution_segmentation": imgsegjoin,
        "segmentation_geometry": seggeom,
        "probability_images": problist
        }



def ljlf_parcellation(
    img,
    segmentation_numbers,
    forward_transforms,
    template,
    templateLabels,
    library_intensity,
    library_segmentation,
    submask_dilation=12,  # a parameter that should be explored
    searcher=1,  # double this for SR
    radder=2,  # double this for SR
    reg_iterations = [100,100,5],
    syn_sampling=2,
    syn_metric='CC',
    output_prefix=None,
    is_test=True,
    sr_model=None,
    verbose=False,
):
    """
    Apply local joint label fusion to an image given a library.

    Arguments
    ---------
    img : ANTsImage
        image to be labeled

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    forward_transforms : list
        transformations that map the template labels to the img

    template : ANTsImages
        a reference template that provides the initial labeling.  could be a
        template from the library or a population-specific template

    templateLabels : ANTsImages
        the reference template segmentation image.

    library_intensity : list of strings or ANTsImages
        the list of library intensity images

    library_segmentation : list of strings or ANTsImages
        the list of library segmentation images

    submask_dilation : integer dilation of mask
        morphological operation that increases the size of the region of interest
        for registration and segmentation

    searcher :  integer search region
        see joint label fusion; this controls the search region

    radder :  integer
        controls the patch radius for similarity calculations

    reg_iterations : list of integers
        controlling the registration iterations; see ants.registration

    syn_sampling : integer
        the metric parameter for registration 2 for CC and 32 or 16 for mattes

    syn_metric : string
        the metric type usually CC or mattes

    output_prefix : string
        the location of the output; should be both a directory and prefix filename

    is_test:  boolean
        set True to set parameters to be faster

    sr_model: string or tensorflow model
        experimental option to upsample before LJLF

    verbose : boolean
        whether to show status updates

    Returns
    -------

    dictionary w/ following key/value pairs:
        `ljlf` : key/value
            the local JLF object

        `segmentation` : ANTsImage
            the output segmentation image

    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    >>> FIXME
    """

    if output_prefix is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_prefix = str(temp_dir) + "LJLF_"
        if verbose:
            print("Created temporary output location: " + output_prefix )


    # build the filenames
    ################################################################################
    if type(library_intensity[0]) == type(str(0)): # these are filenames
        libraryI = []
        for fn in library_intensity:
            libraryI.append(ants.image_read(fn))
        libraryL = []
        for fn in library_segmentation:
            temp = ants.image_read(fn)
            temp = ants.mask_image( temp, temp, segmentation_numbers )
            if not check_for_labels_in_image( segmentation_numbers, temp ):
                warnings.warn( "segmentation_numbers do not exist in" + fn )
            libraryL.append( temp )
    else:
        libraryI = library_intensity
        libraryL = library_segmentation

    ################################################################################
    if not check_for_labels_in_image( segmentation_numbers, templateLabels ):
        warnings.warn( "segmentation_numbers do not exist in templateLabels" )
    initlab0 = ants.apply_transforms(
        img, templateLabels, forward_transforms, interpolator="nearestNeighbor"
    )
    initlab0 = ants.mask_image(initlab0, initlab0, segmentation_numbers)
    ################################################################################
    initlab = initlab0
    # get rid of cerebellum and brain stem
    ################################################################################
    # check outputs at this stage
    ################################################################################
    initlabThresh = ants.threshold_image(initlab, 1, 1e9)
    ################################################################################
    cropmask = ants.morphology(initlabThresh, "dilate", submask_dilation)
    imgc = ants.crop_image(img, cropmask)
    if not sr_model is None: # FIXME replace with an actual model
        newspc = ( np.asarray( ants.get_spacing( imgc ) ) * 0.5 ).tolist()
        imgc = ants.resample_image( imgc, newspc, use_voxels=False, interp_type=0 )
    imgc = ants.iMath(imgc, "TruncateIntensity", 0.001, 0.99999)
    initlabc = ants.resample_image_to_target( initlab, imgc, interp_type="nearestNeighbor"  )
    jlfmask = ants.resample_image_to_target( img*0+1, imgc, interp_type="nearestNeighbor"  )
    mlp1 = False
    deftx = "SyN"
    loctx = "Affine"
    if is_test:
        libraryI = libraryI[0:9]
        libraryL = libraryL[0:9]
        regitsSR = (10, 0, 0)
    ljlf = ants.local_joint_label_fusion(
        target_image=imgc,
        which_labels=segmentation_numbers,
        target_mask=jlfmask,
        initial_label=initlabc,
        type_of_transform=deftx,  # FIXME - try SyN and SyNOnly
        submask_dilation=0,  # we do it this way for consistency across SR and OR
        r_search=searcher,  # should explore 0, 1 and 2
        rad=radder,  # should keep 2 at low-res and search 2 to 4 at high-res
        atlas_list=libraryI,
        label_list=libraryL,
        local_mask_transform=loctx,
        reg_iterations=reg_iterations,
        syn_sampling=syn_sampling,
        syn_metric=syn_metric,
        beta=2,  # higher "sharper" more robust to outliers ( need to check this again )
        rho=0.1,
        nonnegative=True,
        max_lab_plus_one=mlp1,
        verbose=verbose,
        output_prefix=output_prefix,
    )
    ################################################################################
    temp = ants.image_clone(ljlf["ljlf"]["segmentation"], pixeltype="float")
    temp = ants.mask_image( temp, temp, segmentation_numbers )
    hippLabelJLF = ants.resample_image_to_target( temp, img, interp_type="nearestNeighbor" )
    return {
        "ljlf": ljlf,
        "segmentation": hippLabelJLF,
    }




def ljlf_parcellation_one_template(
    img,
    segmentation_numbers,
    forward_transforms,
    template,
    templateLabels,
    templateRepeats,
    submask_dilation=12,  # a parameter that should be explored
    searcher=1,  # double this for SR
    radder=2,  # double this for SR
    reg_iterations = [100,100,5],
    syn_sampling=2,
    syn_metric='CC',
    output_prefix=None,
    verbose=False,
):
    """
    Apply local joint label fusion to an image given a library.

    Arguments
    ---------
    img : ANTsImage
        image to be labeled

    segmentation_numbers : list of target segmentation labels
        list containing integer segmentation labels

    forward_transforms : list
        transformations that map the template labels to the img

    template : ANTsImages
        a reference template that provides the initial labeling.  could be a
        template from the library or a population-specific template

    templateLabels : ANTsImages
        the reference template segmentation image.

    templateRepeats : integer number of registrations to perform
        repeats the template templateRepeats number of times to provide variability

    submask_dilation : integer dilation of mask
        morphological operation that increases the size of the region of interest
        for registration and segmentation

    searcher :  integer search region
        see joint label fusion; this controls the search region

    radder :  integer
        controls the patch radius for similarity calculations

    reg_iterations : list of integers
        controlling the registration iterations; see ants.registration

    syn_sampling : integer
        the metric parameter for registration 2 for CC and 32 or 16 for mattes

    syn_metric : string
        the metric type usually CC or mattes

    output_prefix : string
        the location of the output; should be both a directory and prefix filename

    verbose : boolean
        whether to show status updates

    Returns
    -------

    dictionary w/ following key/value pairs:
        `ljlf` : key/value
            the local JLF object

        `segmentation` : ANTsImage
            the output segmentation image

    Example
    -------
    >>> import ants
    >>> ref = ants.image_read( ants.get_ants_data('r16'))
    >>> FIXME
    """

    if output_prefix is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_prefix = str(temp_dir) + "LJLF_"
        if verbose:
            print("Created temporary output location: " + output_prefix )

    # build the filenames
    ################################################################################
    libraryI = []
    libraryL = []
    for x in range(templateRepeats):
        libraryI.append(template)
        temp = ants.mask_image( templateLabels, templateLabels, segmentation_numbers )
        libraryL.append( temp )

    #  https://mindboggle.readthedocs.io/en/latest/labels.html
    ################################################################################
    initlab0 = ants.apply_transforms(
        img, templateLabels, forward_transforms, interpolator="nearestNeighbor"
    )
    initlab = ants.mask_image(initlab0, initlab0, segmentation_numbers)
    ################################################################################
    if not check_for_labels_in_image( segmentation_numbers, templateLabels ):
        warnings.warn( "segmentation_numbers do not exist in templateLabels" )
    initlabThresh = ants.threshold_image(initlab, 1, 1e9)
    ################################################################################
    cropmask = ants.morphology(initlabThresh, "dilate", submask_dilation)
    imgc = ants.crop_image(img, cropmask)
    imgc = ants.iMath(imgc, "TruncateIntensity", 0.001, 0.99999)
    initlabc = ants.resample_image_to_target( initlab, imgc, interp_type="nearestNeighbor"  )
    jlfmask = ants.resample_image_to_target( img*0+1, imgc, interp_type="nearestNeighbor"  )
    mlp1 = False
    deftx = "SyN"
    loctx = "Affine"
    ljlf = ants.local_joint_label_fusion(
        target_image=imgc,
        which_labels=segmentation_numbers,
        target_mask=jlfmask,
        initial_label=initlabc,
        type_of_transform=deftx,  # FIXME - try SyN and SyNOnly
        submask_dilation=0,  # we do it this way for consistency across SR and OR
        r_search=searcher,  # should explore 0, 1 and 2
        rad=radder,  # should keep 2 at low-res and search 2 to 4 at high-res
        atlas_list=libraryI,
        label_list=libraryL,
        local_mask_transform=loctx,
        reg_iterations=reg_iterations,
        syn_sampling=syn_sampling,
        syn_metric=syn_metric,
        beta=2,  # higher "sharper" more robust to outliers ( need to check this again )
        rho=0.1,
        nonnegative=True,
        max_lab_plus_one=mlp1,
        verbose=verbose,
        output_prefix=output_prefix,
    )
    ################################################################################
    temp = ants.image_clone(ljlf["ljlf"]["segmentation"], pixeltype="float")
    temp = ants.mask_image( temp, temp, segmentation_numbers )
    hippLabelJLF = ants.resample_image_to_target( temp, img, interp_type="nearestNeighbor" )
    return {
        "ljlf": ljlf,
        "segmentation": hippLabelJLF,
    }
