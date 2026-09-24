import unittest

import numpy as np

from scripts.harmony_chroma import extract_cqt_chroma, extract_hpcp


SR = 22050
EXTRACTORS = (extract_cqt_chroma, extract_hpcp)


def tone(frequency: float, seconds: float = 1.5, amplitude: float = 0.3) -> np.ndarray:
    time = np.arange(round(SR * seconds), dtype=np.float32) / SR
    envelope = np.minimum(1.0, np.minimum(time / 0.02, (seconds - time) / 0.02))
    return (amplitude * np.maximum(envelope, 0) * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def chord(frequencies: tuple[float, ...], seconds: float = 1.5) -> np.ndarray:
    return sum((tone(frequency, seconds, 0.2) for frequency in frequencies), np.zeros(round(SR * seconds), np.float32))


def mean_valid(sequence) -> np.ndarray:
    return sequence.chroma[sequence.valid].mean(axis=0)


class HarmonyChromaTest(unittest.TestCase):
    def test_single_note_maps_to_expected_pitch_class(self):
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(tone(440.0), SR)
                self.assertEqual(int(mean_valid(sequence).argmax()), 9)  # A
                self.assertTrue(np.all(np.diff(sequence.timestamps) > 0))

    def test_major_chord_contains_its_three_pitch_classes(self):
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(chord((261.6256, 329.6276, 391.9954)), SR)
                top_three = set(np.argsort(mean_valid(sequence))[-3:])
                self.assertEqual(top_three, {0, 4, 7})  # C, E, G

    def test_minor_chord_contains_its_three_pitch_classes(self):
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(chord((220.0, 261.6256, 329.6276)), SR)
                top_three = set(np.argsort(mean_valid(sequence))[-3:])
                self.assertEqual(top_three, {9, 0, 4})  # A, C, E

    def test_progression_preserves_temporal_order(self):
        audio = np.concatenate([
            chord((261.6256, 329.6276, 391.9954)),
            chord((195.9977, 246.9417, 293.6648)),
        ])
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(audio, SR)
                first = sequence.valid & (sequence.timestamps < 1.2)
                second = sequence.valid & (sequence.timestamps > 1.8)
                self.assertEqual(set(np.argsort(sequence.chroma[first].mean(0))[-3:]), {0, 4, 7})
                self.assertEqual(set(np.argsort(sequence.chroma[second].mean(0))[-3:]), {7, 11, 2})

    def test_transposition_rotates_chroma(self):
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                c_major = mean_valid(extractor(chord((261.6256, 329.6276, 391.9954)), SR))
                d_major = mean_valid(extractor(chord((293.6648, 369.9944, 440.0)), SR))
                similarity = np.corrcoef(np.roll(c_major, 2), d_major)[0, 1]
                self.assertGreater(similarity, 0.95)

    def test_detuned_note_remains_in_same_pitch_class(self):
        detuned_a = 440.0 * 2 ** (25 / 1200)
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(tone(detuned_a), SR)
                self.assertEqual(int(mean_valid(sequence).argmax()), 9)

    def test_silence_is_invalid_not_uniform_chroma(self):
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(np.zeros(SR, dtype=np.float32), SR)
                self.assertFalse(sequence.valid.any())
                self.assertEqual(float(sequence.chroma.sum()), 0.0)

    def test_percussion_has_low_tonal_concentration(self):
        audio = np.zeros(SR * 2, dtype=np.float32)
        pulse = np.hanning(40).astype(np.float32)
        for position in range(0, len(audio) - len(pulse), SR // 4):
            audio[position:position + len(pulse)] = pulse
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(audio, SR)
                if sequence.valid.any():
                    self.assertLess(float(sequence.tonal_concentration[sequence.valid].mean()), 0.15)
                else:
                    self.assertEqual(float(sequence.chroma.sum()), 0.0)

    def test_short_clip_is_supported(self):
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                sequence = extractor(tone(440.0, seconds=0.1), SR)
                self.assertTrue(sequence.valid.any())
                self.assertEqual(int(mean_valid(sequence).argmax()), 9)

    def test_nonfinite_audio_is_rejected(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[10] = np.nan
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    extractor(audio, SR)

    def test_output_is_deterministic(self):
        audio = chord((196.0, 246.9417, 293.6648), seconds=0.75)
        for extractor in EXTRACTORS:
            with self.subTest(extractor=extractor.__name__):
                first = extractor(audio, SR)
                second = extractor(audio, SR)
                np.testing.assert_array_equal(first.chroma, second.chroma)
                np.testing.assert_array_equal(first.valid, second.valid)
                self.assertEqual(first.tuning_value, second.tuning_value)
                self.assertEqual(first.tuning_unit, second.tuning_unit)


if __name__ == "__main__":
    unittest.main()
