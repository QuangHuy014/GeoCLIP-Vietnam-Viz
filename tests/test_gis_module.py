# ==============================================================================
# UNIT TESTS & INTEGRATION TESTS FOR GIS MODULE (TASK 2.1)
# ==============================================================================

import os
import tempfile
import unittest
import torch

from src.gis.location_encoder import equal_earth_projection, StandaloneLocationEncoder
from src.gis.rff_layers import GaussianEncoding, MultiScaleGaussianEncoding


class TestEqualEarthProjection(unittest.TestCase):
    """Test cases for Equal Earth Projection."""

    def test_single_coordinate_tphcm(self):
        xy = equal_earth_projection(10.795, 106.721)
        self.assertIsInstance(xy, torch.Tensor)
        self.assertEqual(xy.shape, (2,))
        self.assertFalse(torch.isnan(xy).any())
        self.assertFalse(torch.isinf(xy).any())

    def test_single_coordinate_hanoi(self):
        xy = equal_earth_projection(21.028, 105.834)
        self.assertIsInstance(xy, torch.Tensor)
        self.assertEqual(xy.shape, (2,))
        self.assertFalse(torch.isnan(xy).any())
        self.assertFalse(torch.isinf(xy).any())

    def test_batch_coordinates(self):
        coords = torch.tensor(
            [
                [10.795, 106.721],
                [21.028, 105.834],
                [16.054, 108.202],
            ],
            dtype=torch.float32,
        )
        xy = equal_earth_projection(coords)
        self.assertEqual(xy.shape, (3, 2))
        self.assertFalse(torch.isnan(xy).any())
        self.assertFalse(torch.isinf(xy).any())

    def test_invalid_latitude_raises(self):
        with self.assertRaises(ValueError):
            equal_earth_projection(95.0, 106.721)

        with self.assertRaises(ValueError):
            equal_earth_projection(-100.0, 106.721)

    def test_invalid_longitude_raises(self):
        with self.assertRaises(ValueError):
            equal_earth_projection(10.795, 190.0)

        with self.assertRaises(ValueError):
            equal_earth_projection(10.795, -200.0)


class TestMultiscaleGaussianEncoding(unittest.TestCase):
    """Test cases for GaussianEncoding and MultiScaleGaussianEncoding (RFF)."""

    def test_gaussian_encoding_init(self):
        layer = GaussianEncoding(sigma=16.0, input_size=2, encoded_size=256)
        self.assertIsInstance(layer, torch.nn.Module)
        self.assertTrue(hasattr(layer, "B"))
        self.assertEqual(layer.B.shape, (2, 256))

    def test_multiscale_sigmas(self):
        multi_rff = MultiScaleGaussianEncoding(sigmas=[1.0, 16.0, 256.0])
        self.assertEqual(len(multi_rff.sigmas), 3)
        self.assertEqual(multi_rff.sigmas, [1.0, 16.0, 256.0])
        self.assertEqual(multi_rff.out_dim, 3 * (2 * 256))  # 1536

    def test_rff_forward_single_and_batch(self):
        multi_rff = MultiScaleGaussianEncoding(sigmas=[1.0, 16.0, 256.0])
        single_in = torch.tensor([0.5, 0.5])
        out_single = multi_rff(single_in)
        self.assertEqual(out_single.shape, (1536,))
        self.assertFalse(torch.isnan(out_single).any())

        batch_in = torch.tensor([[0.5, 0.5], [1.0, -1.0]])
        out_batch = multi_rff(batch_in)
        self.assertEqual(out_batch.shape, (2, 1536))
        self.assertFalse(torch.isnan(out_batch).any())

    def test_fourier_matrix_persistence(self):
        """Verify that B buffer is NOT regenerated during forward calls."""
        layer = GaussianEncoding(sigma=1.0)
        B_initial = layer.B.clone()

        v = torch.tensor([[0.1, 0.2]])
        _ = layer(v)
        _ = layer(v)

        # Buffer B should remain exactly identical
        self.assertTrue(torch.equal(layer.B, B_initial))

    def test_rff_stability(self):
        multi_rff = MultiScaleGaussianEncoding(sigmas=[1.0, 16.0, 256.0], seed=123)
        v = torch.tensor([[0.123, 0.456]])
        out1 = multi_rff(v)
        out2 = multi_rff(v)
        self.assertTrue(torch.equal(out1, out2))


class TestStandaloneLocationEncoder(unittest.TestCase):
    """Test cases for StandaloneLocationEncoder."""

    def test_single_coordinate_output_shape(self):
        encoder = StandaloneLocationEncoder()

        # TP.HCM
        out_hcm = encoder([[10.795, 106.721]])
        self.assertEqual(out_hcm.shape, (1, 512))
        self.assertFalse(torch.isnan(out_hcm).any())
        self.assertFalse(torch.isinf(out_hcm).any())

        # Hà Nội
        out_hanoi = encoder([[21.028, 105.834]])
        self.assertEqual(out_hanoi.shape, (1, 512))
        self.assertFalse(torch.isnan(out_hanoi).any())
        self.assertFalse(torch.isinf(out_hanoi).any())

    def test_batch_output_shape(self):
        encoder = StandaloneLocationEncoder()
        batch_coords = torch.tensor(
            [
                [10.795, 106.721],
                [21.028, 105.834],
                [16.054, 108.202],
            ]
        )
        out = encoder(batch_coords)
        self.assertEqual(out.shape, (3, 512))
        self.assertFalse(torch.isnan(out).any())

    def test_load_weights_and_stability(self):
        encoder1 = StandaloneLocationEncoder()
        
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save state dict
            torch.save(encoder1.state_dict(), tmp_path)

            encoder2 = StandaloneLocationEncoder()
            success = encoder2.load_pretrained_weights(tmp_path)
            self.assertTrue(success)

            coords = torch.tensor([[10.795, 106.721]])
            out1 = encoder1(coords)
            out2 = encoder2(coords)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-6))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_missing_weights_file_raises(self):
        encoder = StandaloneLocationEncoder()
        with self.assertRaises(FileNotFoundError):
            encoder.load_pretrained_weights("non_existent_weights_file.pth")

    def test_invalid_coordinates_raises(self):
        encoder = StandaloneLocationEncoder()
        with self.assertRaises(ValueError):
            encoder([[999.0, 106.721]])


class TestIntegration(unittest.TestCase):
    """End-to-End integration test of GIS Location Encoder Pipeline."""

    def test_pipeline_integration(self):
        encoder = StandaloneLocationEncoder()
        test_points = torch.tensor(
            [
                [10.795, 106.721],  # TP.HCM
                [21.028, 105.834],  # Hà Nội
                [16.054, 108.202],  # Đà Nẵng
            ]
        )
        embedding = encoder(test_points)
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, (3, 512))
        self.assertFalse(torch.isnan(embedding).any())
        self.assertFalse(torch.isinf(embedding).any())


if __name__ == "__main__":
    unittest.main()
