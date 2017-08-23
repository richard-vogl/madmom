# encoding: utf-8
# pylint: skip-file
"""
This file contains tests for the madmom.features.drums module.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os.path import join as pj

from . import AUDIO_PATH, ACTIVATIONS_PATH, DETECTIONS_PATH

from madmom.features import Activations
from madmom.features.drums import *

sample_file = pj(AUDIO_PATH, 'sample.wav')
sample_act = Activations(pj(ACTIVATIONS_PATH, 'sample.drums_crnn.npz'))


class TestDrumProcessorClass(unittest.TestCase):
    def setUp(self):
        self.processor = CRNNDrumProcessor()

    def test_process(self):
        act = self.processor(sample_file)
        self.assertTrue(np.allclose(act, sample_act))


class TestDrumPeakPickingProcessorClass(unittest.TestCase):

    def setUp(self):
        self.processor = DrumPeakPickingProcessor(fps=sample_act.fps)

    def test_process(self):
        drums = self.processor(sample_act)
        self.assertTrue(np.allclose(drums, [[0.13, 0], [0.13, 2], [0.48, 2],
                                            [0.65, 0], [0.80, 0], [0.84, 1],
                                            [1.16, 0], [1.16, 2], [1.66, 1],
                                            [1.84, 0], [1.84, 2], [2.18, 1],
                                            [2.18, 2], [2.70, 0]]))
