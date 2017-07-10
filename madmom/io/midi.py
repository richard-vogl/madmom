# encoding: utf-8
# pylint: disable=no-member
"""
This module contains MIDI functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import mido

DEFAULT_TEMPO = 500000
DEFAULT_TIME_SIGNATURE = (4, 4)
DEFAULT_TICKS_PER_BEAT = 480


# TODO: functions copied and corrected from mido, should go upstream
def second2tick(second, ticks_per_beat, tempo, time_signature):
    """Convert absolute time in seconds to ticks."""
    return int(second / tempo * 1e-6 / ticks_per_beat * time_signature[1] / 4.)


def tick2second(tick, ticks_per_beat, tempo, time_signature):
    """Convert absolute time in ticks to seconds."""
    return tick * tempo * 1e-6 / ticks_per_beat * time_signature[1] / 4.


def bpm2tempo(bpm, time_signature):
    """Convert beats per minute (BPM) to MIDI file tempo."""
    return int(60 * 1e6 / bpm * time_signature[1] / 4.)


def tempo2bpm(tempo, time_signature):
    """Convert MIDI file tempo to beats per minute (BPM)."""
    # One minute is 60 million microseconds.
    return 60 * 1e6 / tempo * time_signature[1] / 4.


def beats2ticks(beats, ticks_per_beat, time_signature):
    """Convert beats to ticks."""
    return int(ticks_per_beat * time_signature[1] / 4. / beats)


def ticks2beats(tick, ticks_per_beat, time_signature):
    """Convert ticks to beats."""
    return tick / ticks_per_beat * time_signature[1] / 4.


class MIDIFile(mido.MidiFile):
    """
    MIDI File.

    Examples
    --------
    Create a MIDI file from an array with notes. The format of the note array
    is: 'onset time', 'pitch', 'duration', 'velocity', 'channel'. The last
    column can be omitted, assuming channel 0.

    >>> notes = np.array([[0, 50, 1, 60], [0.5, 62, 0.5, 90]])
    >>> m = MIDIFile.from_notes(notes)
    >>> m  # doctest: +ELLIPSIS
    <madmom.io.midi.MIDIFile object at 0x...>

    The notes can be accessed as a numpy array in various formats (default is
    seconds):

    >>> m.notes
    array([[  0. ,  50. ,   1. ,  60. ,   0. ],
           [  0.5,  62. ,   0.5,  90. ,   0. ]])
    >>> m.unit ='ticks'
    >>> m.notes
    array([[   0.,   50.,  960.,   60.,    0.],
           [ 480.,   62.,  480.,   90.,    0.]])
    >>> m.unit = 'seconds'
    >>> m.notes
    array([[  0. ,  50. ,   1. ,  60. ,   0. ],
           [  0.5,  62. ,   0.5,  90. ,   0. ]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[  0.,  50.,   2.,  60.,   0.],
           [  1.,  62.,   1.,  90.,   0.]])

    >>> m = MIDIFile.from_notes(notes, tempo=60)
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[   0.,   50.,  480.,   60.,    0.],
           [ 240.,   62.,  240.,   90.,    0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[  0. ,  50. ,   1. ,  60. ,   0. ],
           [  0.5,  62. ,   0.5,  90. ,   0. ]])

    >>> m = MIDIFile.from_notes(notes, tempo=60, time_signature=(2, 2))
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[   0.,   50.,  960.,   60.,    0.],
           [ 480.,   62.,  480.,   90.,    0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[  0. ,  50. ,   1. ,  60. ,   0. ],
           [  0.5,  62. ,   0.5,  90. ,   0. ]])

    >>> m = MIDIFile.from_notes(notes, tempo=240, time_signature=(3, 8))
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[   0.,   50.,  960.,   60.,    0.],
           [ 480.,   62.,  480.,   90.,    0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[  0.,  50.,   4.,  60.,   0.],
           [  2.,  62.,   2.,  90.,   0.]])

    """

    unit = 's'
    timing = 'abs'

    def __iter__(self):
        # Note: overwrite the iterator provided by mido to be able to set the
        #       unit of the notes etc. and if they are returned with relative
        #       or absolute timing.
        # The tracks of type 2 files are not in sync, so they can
        # not be played back like this.
        if self.type == 2:
            raise TypeError("can't merge tracks in type 2 (asynchronous) file")

        tempo = DEFAULT_TEMPO
        time_signatue = DEFAULT_TIME_SIGNATURE
        cum_time = 0
        for msg in mido.merge_tracks(self.tracks):
            # Convert message time to desired unit.
            if msg.time > 0:
                if self.unit.lower() in ('t', 'ticks'):
                    time = msg.time
                elif self.unit.lower() in ('s', 'sec', 'seconds'):
                    time = tick2second(msg.time, self.ticks_per_beat, tempo,
                                       time_signatue)
                elif self.unit.lower() in ('b', 'beats'):
                    time = ticks2beats(msg.time, self.ticks_per_beat,
                                       time_signatue)
                else:
                    raise ValueError("`unit` must be either 'ticks', 't', "
                                     "'seconds', 's', 'beats', 'b', not %s." %
                                     self.unit)
            else:
                time = 0
            # Convert relative time to absolute values if needed.
            if self.timing.lower() in ('a', 'abs', 'absolute'):
                cum_time += time
            elif self.timing.lower() in ('r', 'rel', 'relative'):
                cum_time = time
            else:
                raise ValueError("`timing` must be either 'relative', 'rel', "
                                 "'r', or 'absolute', 'abs', 'a', not %s." %
                                 self.timing)

            yield msg.copy(time=cum_time)

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'time_signature':
                time_signatue = (msg.numerator, msg.denominator)

    def __repr__(self):
        return object.__repr__(self)

    @property
    def tempi(self):
        """
        Tempi (mircoseconds per beat) of the MIDI file.

        Returns
        -------
        tempi : numpy array
            Array with tempi (time, tempo).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
        # list for all tempi
        tempi = []
        # process all events
        for msg in self:
            if msg.type == 'set_tempo':
                tempi.append((msg.time, msg.tempo))
        # make sure a tempo is set (and occurs at time 0)
        if not tempi or tempi[0][0] > 0:
            tempi.insert(0, (0, DEFAULT_TEMPO))
        # tempo is given in microseconds per quarter note
        # TODO: add otption to return in BPM
        return np.asarray(tempi, np.float)

    @property
    def time_signatures(self):
        """
        Time signatures of the MIDI file.

        Returns
        -------
        time_signatures : numpy array
            Array with time signatures (time, numerator, denominator).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
        # list for all tempi
        signatures = []
        # process all events
        for msg in self:
            if msg.type == 'time_signature':
                signatures.append((msg.time, msg.numerator, msg.denominator))
        # make sure a signatures is set (and occurs at time 0)
        if not signatures or signatures[0][0] > 0:
            signatures.insert(0, (0, DEFAULT_TIME_SIGNATURE[0],
                                  DEFAULT_TIME_SIGNATURE[1]))
        # return time signatures
        return np.asarray(signatures, dtype=np.float)

    @property
    def notes(self):
        """
        Notes of the MIDI file.

        Returns
        -------
        notes : numpy array
            Array with notes (onset time, pitch, duration, velocity, channel).

        """
        # list for all notes
        notes = []
        # dictionary for storing the last onset time and velocity for each
        # individual note (i.e. same pitch and channel)
        sounding_notes = {}

        # as key for the dict use channel * 128 (max number of pitches) + pitch
        def note_hash(channel, pitch):
            """Generate a note hash."""
            return channel * 128 + pitch

        # process all events
        for msg in self:
            # use only note on or note off events
            note_on = msg.type == 'note_on'
            note_off = msg.type == 'note_off'
            # hash sounding note
            if note_on or note_off:
                note = note_hash(msg.channel, msg.note)
            # if it's a note on event with a velocity > 0,
            if note_on and msg.velocity > 0:
                # save the onset time and velocity
                sounding_notes[note] = (msg.time, msg.velocity)
            # if it's a note off or a note on event with a velocity of 0,
            elif note_off or (note_on and msg.velocity == 0):
                if note not in sounding_notes:
                    raise RuntimeError("ignoring %s" % msg)
                    continue
                # append the note to the list
                notes.append((sounding_notes[note][0], msg.note,
                              msg.time - sounding_notes[note][0],
                              sounding_notes[note][1], msg.channel))
                # remove hash from dict
                del sounding_notes[note]

        # sort the notes and convert to numpy array
        return np.asarray(sorted(notes), dtype=np.float)

    @classmethod
    def from_notes(cls, notes, tempo=DEFAULT_TEMPO,
                   time_signature=DEFAULT_TIME_SIGNATURE,
                   ticks_per_beat=DEFAULT_TICKS_PER_BEAT):
        """
        Create a MIDIFile from the given notes.

        Parameters
        ----------
        notes : numpy array
            Array with notes, one per row. The columns are defined as:
            (onset time, pitch, duration, velocity, [channel]).
        tempo : float, optional
            Tempo of the MIDI track, given in bpm or microseconds per beat.
            The unit is determined automatically by the value:

            - `tempo` <= 1000: bpm
            - `tempo` > 1000: microseconds per beat

        time_signature : tuple, optional
            Time signature of the track, e.g. (4, 4) for 4/4.
        ticks_per_beat : int
            Resolution (i.e. ticks per beat) of the MIDI file.

        Returns
        -------
        :class:`MIDIFile` instance
            :class:`MIDIFile` instance with all notes collected in one track.

        Notes
        -----
        All note events (including the generated tempo and time signature
        events) are written into a single track (i.e. MIDI file format 0).

        """
        # create new MIDI file
        midi_file = cls(type=0, ticks_per_beat=ticks_per_beat)
        # convert tempo
        if tempo <= 1000:
            tempo = bpm2tempo(tempo, time_signature)
        # create new track and add tempo and time signature information
        track = midi_file.add_track()
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        track.append(mido.MetaMessage('time_signature',
                                      numerator=time_signature[0],
                                      denominator=time_signature[1]))
        # create note on/off messages with absolute timing
        messages = []
        for note in notes:
            try:
                onset, pitch, duration, velocity, channel = note
            except ValueError:
                onset, pitch, duration, velocity = note
                channel = 0
            pitch = int(pitch)
            velocity = int(velocity)
            offset = onset + duration

            onset = int(mido.second2tick(onset, ticks_per_beat, tempo))
            note_on = mido.Message('note_on', time=onset, note=pitch,
                                   velocity=velocity, channel=channel)
            offset = int(mido.second2tick(offset, ticks_per_beat, tempo))
            note_off = mido.Message('note_off', time=offset, note=pitch,
                                    channel=channel)
            messages.extend([note_on, note_off])
        # sort them, convert to relative timing and append to track
        messages.sort(key=lambda msg: msg.time)
        messages = mido.midifiles.tracks._to_reltime(messages)
        track.extend(messages)
        # return MIDI file
        return midi_file

    def save(self, filename):
        """
        Save to MIDI file.

        Parameters
        ----------
        filename : str or open file handle
            The MIDI file name.

        """
        # if we get a filename, open the file
        if not hasattr(filename, 'write'):
            filename = open(filename, 'wb')
        # write the MIDI stream
        self._save(filename)


def process_notes(data, output=None):
    """
    This is a simple processing function. It either loads the notes from a MIDI
    file and or writes the notes to a file.

    The behaviour depends on the presence of the `output` argument, if 'None'
    is given, the notes are read, otherwise the notes are written to file.

    Parameters
    ----------
    data : str or numpy array
        MIDI file to be loaded (if `output` is 'None') / notes to be written.
    output : str, optional
        Output file name. If set, the notes given by `data` are written.

    Returns
    -------
    notes : numpy array
        Notes read/written.

    """
    # TODO: copied from madmom.utils.midi, refactor? See issue #302
    if output is None:
        # load the notes
        return MIDIFile(data).notes
    # output notes
    MIDIFile.from_notes(data).write(output)
    return data
