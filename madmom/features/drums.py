# encoding: utf-8
# pylint: disable=invalid-name
"""
This module contains drum transcription related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from .notes import NotePeakPickingProcessor
from ..processors import SequentialProcessor, ParallelProcessor
from ..ml.nn import average_predictions


class PadProcessor:

    def __init__(self, pad):
        self.pad = pad

    def __call__(self, data):
      """
      Pad the data by repeating the first and last frame [pad] times.

      Parameters
      ----------
      data: numpy array
          Input data.

      pad: int
          Number of repetitions for first and last frame

      Returns
      -------
      numpy array
          Padded data.

      """

      pad_start = np.repeat(data[:1], self.pad, axis=0)
      pad_stop = np.repeat(data[-1:], self.pad, axis=0)
      return np.concatenate((pad_start, data, pad_stop))


def _crnn_drum_processor_stack(data):
    """
    Stacks a row of zeros between the spctrogram and the differences.

    Parameters
    ----------
    data : tuple
        Two numpy arrays (spectrogram, differences).

    Returns
    -------
    numpy array
        Stacked input with 0's in between.

    """
    return np.hstack((data[0], np.zeros((data[0].shape[0], 1)), data[1]))


def _make_preprocessor(settings, pad):
    from ..audio.spectrogram import (
        LogarithmicFilteredSpectrogramProcessor,
        SpectrogramDifferenceProcessor)
    from ..audio.filters import LogarithmicFilterbank
    from ..audio.signal import SignalProcessor, FramedSignalProcessor
    from ..audio.stft import ShortTimeFourierTransformProcessor

    sig = SignalProcessor(num_channels=1, sample_rate=settings['sample_rate'])
    frames = FramedSignalProcessor(frame_size=settings['frame_size'], fps=settings['fps'])
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    spec = LogarithmicFilteredSpectrogramProcessor(
        num_channels=1, sample_rate=settings['sample_rate'],
        filterbank=LogarithmicFilterbank, frame_size=settings['frame_size'], fps=settings['fps'],
        num_bands=settings['num_bands'], fmin=settings['fmin'], fmax=settings['fmax'],
        norm_filters=settings['norm_filters'])
    if settings['diff']:
        if 'pad' in settings and settings['pad']:
            stack = _crnn_drum_processor_stack
        else:
            stack = np.hstack
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True,
            stack_diffs=stack)
        # process input data
        if pad > 0:
            pre_processor = SequentialProcessor(
                (sig, frames, stft, spec, diff, PadProcessor(pad)))
        else:
            pre_processor = SequentialProcessor((sig, frames, stft, spec, diff))

    else:
        if pad > 0:
            pre_processor = SequentialProcessor(
                (sig, frames, stft, spec, PadProcessor(pad)))
        else:
            pre_processor = SequentialProcessor((sig, frames, stft, spec))

    return pre_processor


def _flatten_pred(predictions):
    predictions_flat = []
    # average predictions if needed
    for pred in predictions:
        if type(pred) == list:
            for subpred in pred:
                predictions_flat.append(subpred)
        else:
            predictions_flat.append(pred)
    return predictions_flat


MIREX17_B_SET = {
    'name': "feat_mirex17_b",
    'fps': 100,
    'fmin': 30,
    'fmax': 15000,
    'frame_size': 2048,
    'sample_rate': 44100,
    'num_bands': 12,
    'norm_filters': True,
    'start_silence': 0.25,
    'target_shift': 0.00,
    'soft_target': 0,
    'diff': False,
    'pad': False,
}

ISMIR17CNN_SET = {
    'name': "feat_ismir17cnn",
    'fps': 100,
    'fmin': 20,
    'fmax': 20000,
    'frame_size': 2048,
    'sample_rate': 44100,
    'num_bands': 12,
    'norm_filters': True,
    'start_silence': 0.25,
    'target_shift': 0.00,
    'soft_target': 0,
    'diff': True,
}

CRNN_MODEL = 'CRNN5'
CNN_MODEL = 'CNN3'
BRNN_MODEL = 'BRNN2'
SUPER_MODEL = 'ENS'


class CRNNDrumProcessor(SequentialProcessor):
    """

    """

    def __init__(self, **kwargs):
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DRUMS_CRNN, DRUMS_BRNN, DRUMS_CNN, DRUMS_BRNN_R, DRUMS_CNN_R, DRUMS_CRNN_R

        models = {
            CRNN_MODEL: {
                'settings': MIREX17_B_SET,
                'pad': 6,
                'model_file': DRUMS_CRNN,
                'model_file_rand': DRUMS_CRNN_R,
            },
            CNN_MODEL: {
                'settings': MIREX17_B_SET,
                'pad': 7,
                'model_file': DRUMS_CNN,
                'model_file_rand': DRUMS_CNN_R,
            },
            BRNN_MODEL: {
                'settings': ISMIR17CNN_SET,
                'pad': 0,
                'model_file': DRUMS_BRNN,
                'model_file_rand': DRUMS_BRNN_R,
            }
        }

        if 'model' in kwargs:
            model_name = kwargs['model']
        else:
            model_name = CRNN_MODEL
        if 'rand_model' in kwargs:
            model_rand = kwargs['rand_model']
        else:
            model_rand = False

        if model_name == SUPER_MODEL:
            # cnn part
            nn_crnn = NeuralNetworkEnsemble.load(DRUMS_CRNN)
            nn_cnn = NeuralNetworkEnsemble.load(DRUMS_CNN)

            cnn_pre_processor = _make_preprocessor(MIREX17_B_SET, 0)
            crnn_processor = SequentialProcessor((PadProcessor(6), nn_crnn))
            cnn_processor = SequentialProcessor((PadProcessor(7), nn_cnn))

            cnn_paralell_processor = ParallelProcessor((crnn_processor, cnn_processor))
            two_cnn_processor = SequentialProcessor((cnn_pre_processor, cnn_paralell_processor))

            # brnn part
            brnn_pre_processor = _make_preprocessor(ISMIR17CNN_SET, 0)
            nn_brnn = NeuralNetworkEnsemble.load(DRUMS_BRNN)

            brnn_processor = SequentialProcessor((brnn_pre_processor, nn_brnn))

            nn_ens = ParallelProcessor((two_cnn_processor, brnn_processor))

            super(CRNNDrumProcessor, self).__init__((nn_ens, _flatten_pred, average_predictions))

        else:
            model_dict = models[model_name]
            settings = model_dict['settings']
            pad = model_dict['pad']
            if model_rand:
                model_file = model_dict['model_file_rand']
            else:
                model_file = model_dict['model_file']

            # signal processing chain
            pre_processor = _make_preprocessor(settings, pad)
            # process with a NN
            nn = NeuralNetworkEnsemble.load(model_file)
            # instantiate a SequentialProcessor
            super(CRNNDrumProcessor, self).__init__((pre_processor, nn))

    @staticmethod
    def add_arguments(parser, model=CRNN_MODEL):
        """
        Add drum NN related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        model : string
            Model name to initialize drum transcription NN with.

        Returns
        -------
        parser_group : argparse argument group
            Drum transcription neural network argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add onset peak-picking related options to the existing parser
        g = parser.add_argument_group('drum-transcription arguments')
        g.add_argument('-m', dest='model', action='store', default=model,
                       help='NN model to be used for transcription ('+BRNN_MODEL+', '+CNN_MODEL+', or '+CRNN_MODEL+') [default=%(default)s]')

        g.add_argument('--rand', dest='rand_model', action='store_true', default=False,
                       help='Use models trained on randomized data splits.')

        # return the argument group so it can be modified if needed
        return g


class DrumPeakPickingProcessor(NotePeakPickingProcessor):
    """
    This class implements the drum peak-picking functionality.

    Parameters
    ----------
    threshold : float
        Threshold for peak-picking.
    smooth : float, optional
        Smooth the activation function over `smooth` seconds.
    pre_avg : float, optional
        Use `pre_avg` seconds past information for moving average.
    post_avg : float, optional
        Use `post_avg` seconds future information for moving average.
    pre_max : float, optional
        Use `pre_max` seconds past information for moving maximum.
    post_max : float, optional
        Use `post_max` seconds future information for moving maximum.
    combine : float, optional
        Only report one drum hit per instrument within `combine` seconds.
    delay : float, optional
        Report the detected drums `delay` seconds delayed.
    online : bool, optional
        Use online peak-picking, i.e. no future information.
    fps : float, optional
        Frames per second used for conversion of timings.

    Returns
    -------
    drums : numpy array
        Detected drums [seconds, pitch].

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), `pre_avg` and
    `post_avg` should be set to 0.
    For peak picking of local maxima, set `pre_max` >= 1. / `fps` and
    `post_max` >= 1. / `fps`.
    For online peak picking, all `post_` parameters are set to 0.

    Examples
    --------
    Create a DrumPeakPickingProcessor. The returned array represents the note
    positions in seconds, thus the expected sampling rate has to be given.

    >>> proc = DrumPeakPickingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.drums.DrumPeakPickingProcessor object at 0x...>

    Call this DrumPeakPickingProcessor with the drum activations from a
    CRNNDrumProcessor.

    >>> act = CRNNDrumProcessor()('tests/data/audio/stereo_sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[0.13, 0.],
          [0.13, 2.],
          [0.48, 2.],
          [0.65, 0.],
          [0.8, 0.],
          [1.16, 0.],
          [1.16, 2.],
          [1.52, 0.],
          [1.66, 1.],
          [1.84, 0.],
          [1.84, 2.],
          [2.18, 1.],
          [2.7, 0.]])

    """

    pitch_offset = 0
