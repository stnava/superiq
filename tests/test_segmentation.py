import sys
import os
import argparse
import unittest
import warnings
import contextlib
import numpy as np
import numpy.testing as nptest

import unittest
import ants
import superiq

def run_tests():
    unittest.main()

class TestModule_ljlf_parcellation_one_template(unittest.TestCase):
    def test_ljlf_parcellation_one_template_segmentation_isin(self):
        tar = ants.image_read( ants.get_ants_data('r16'))
        ref = ants.image_read( ants.get_ants_data('r27'))
        refseg = ants.kmeans_segmentation( ref, k=4, kmask=None, mrf=0 )['segmentation']
        fwd = ants.registration( tar, ref, 'SyN' )['fwdtransforms']
        tarlab = [4,3]
        temp = superiq.ljlf_parcellation_one_template( tar, tarlab,
          fwd, ref, refseg,
          templateRepeats=2, submask_dilation=4, verbose=True)
        ulab = temp['segmentation'].unique()
        testit = (int(ulab[1]) in tarlab) & (int(ulab[2]) in tarlab)
        self.assertTrue( testit )

class TestModule_ljlf_parcellation(unittest.TestCase):
    def test_ljlf_parcellation_segmentation_isin(self):
        tar = ants.image_read( ants.get_ants_data('r16'))
        ref1 = ants.image_read( ants.get_ants_data('r27'))
        ref2 = ants.image_read( ants.get_ants_data('r64'))
        refseg1 = ants.kmeans_segmentation( ref1, k=4, kmask=None, mrf=0 )['segmentation']
        refseg2 = ants.kmeans_segmentation( ref2, k=4, kmask=None, mrf=0 )['segmentation']
        fwd = ants.registration( tar, ref1, 'SyN' )['fwdtransforms']
        tarlab = [4,3]
        temp = superiq.ljlf_parcellation(
          tar, tarlab,
          fwd, ref1, refseg1,
          [ref1,ref2], [refseg1,refseg2], submask_dilation=4, verbose=True)
        ulab = temp['segmentation'].unique()
        testit = (int(ulab[1]) in tarlab) & (int(ulab[2]) in tarlab)
        self.assertTrue( testit )

if __name__ == "__main__":
    run_tests()
