import unittest

import numpy as np

from processor import calculate_caf, extract_radar_peaks


class ProcessorTests(unittest.TestCase):
    def test_calculate_caf_and_extract_radar_peaks(self):
        sample_rate = 12000.0
        length = 512
        rng = np.random.default_rng(0)
        reference = rng.standard_normal(length)
        received = reference + 0.05 * rng.standard_normal(length)

        caf_matrix, doppler_axis = calculate_caf(received, reference, sample_rate=sample_rate)
        delay, doppler, peak_row = extract_radar_peaks(caf_matrix, doppler_axis, max_delay_samples=128)

        self.assertEqual(caf_matrix.shape[0], len(doppler_axis))
        self.assertEqual(caf_matrix.shape[1], length)
        self.assertGreaterEqual(delay, 0)
        self.assertLess(delay, 128)
        self.assertEqual(len(peak_row), 121)
        self.assertTrue(np.isfinite(doppler))


if __name__ == "__main__":
    unittest.main()
