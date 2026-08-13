# ==============================================================================
# UNIT TESTS FOR GIS DISTANCE ERROR METRICS (TASK 2.3)
# ==============================================================================

import unittest
import torch

from src.gis.distance_metrics import (
    calculate_geodesic_distance,
    calculate_batch_haversine_distance,
    compute_distance_accuracy_metrics,
    haversine_distance,
)


class TestDistanceMetrics(unittest.TestCase):
    """Unit tests for GIS distance metrics and Acc@K accuracy evaluation."""

    def test_identical_points_return_zero_distance(self):
        point = (10.795, 106.721)
        dist = calculate_geodesic_distance(point, point)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_known_distance_hanoi_tphcm(self):
        hanoi = (21.028, 105.834)
        tphcm = (10.795, 106.721)

        dist_haversine = calculate_geodesic_distance(hanoi, tphcm, method="haversine")
        dist_geodesic = calculate_geodesic_distance(hanoi, tphcm, method="geodesic")

        # Distance between Hanoi and HCMC is approx 1130 km
        self.assertGreater(dist_haversine, 1100.0)
        self.assertLess(dist_haversine, 1170.0)

        self.assertGreater(dist_geodesic, 1100.0)
        self.assertLess(dist_geodesic, 1170.0)

    def test_invalid_coordinates_raise(self):
        with self.assertRaises(ValueError):
            haversine_distance((95.0, 106.721), (10.795, 106.721))

        with self.assertRaises(ValueError):
            haversine_distance((10.795, 106.721), (10.795, 200.0))

    def test_batch_distance_calculation(self):
        targets = [
            (10.795, 106.721),  # TP.HCM
            (21.028, 105.834),  # Hà Nội
            (16.054, 108.202),  # Đà Nẵng
        ]
        predictions = [
            (10.790, 106.725),  # Lệch nhỏ (~0.7 km)
            (21.030, 105.830),  # Lệch nhỏ (~0.5 km)
            (10.795, 106.721),  # Lệch lớn (~600 km)
        ]

        distances = calculate_batch_haversine_distance(targets, predictions)
        self.assertEqual(len(distances), 3)
        self.assertLess(distances[0], 2.0)
        self.assertLess(distances[1], 2.0)
        self.assertGreater(distances[2], 500.0)

    def test_torch_tensor_batch_input(self):
        targets_tensor = torch.tensor([[10.795, 106.721], [21.028, 105.834]])
        preds_tensor = torch.tensor([[10.795, 106.721], [21.028, 105.834]])

        distances = calculate_batch_haversine_distance(targets_tensor, preds_tensor)
        self.assertEqual(len(distances), 2)
        self.assertAlmostEqual(distances[0], 0.0, places=4)
        self.assertAlmostEqual(distances[1], 0.0, places=4)

    def test_compute_distance_accuracy_metrics(self):
        targets = [
            (10.795, 106.721),  # TP.HCM
            (21.028, 105.834),  # Hà Nội
            (16.054, 108.202),  # Đà Nẵng
            (20.910, 107.183),  # Hạ Long
        ]
        predictions = [
            (10.795, 106.721),  # Lệch 0km -> Acc@1km, Acc@25km, etc.
            (10.795, 106.721),  # Lệch ~1130km -> Acc@2500km
            (16.060, 108.205),  # Lệch ~0.7km -> Acc@1km
            (20.920, 107.200),  # Lệch ~2.0km -> Acc@25km
        ]

        metrics = compute_distance_accuracy_metrics(targets, predictions, thresholds_km=[1, 25, 200, 750, 2500])

        self.assertIn("Acc@1km", metrics)
        self.assertIn("Acc@25km", metrics)
        self.assertIn("Acc@200km", metrics)
        self.assertIn("Acc@750km", metrics)
        self.assertIn("Acc@2500km", metrics)
        self.assertIn("mean_error_km", metrics)
        self.assertIn("median_error_km", metrics)

        # 2 out of 4 (50%) are within 1km
        self.assertEqual(metrics["Acc@1km"], 50.0)
        # 3 out of 4 (75%) are within 25km
        self.assertEqual(metrics["Acc@25km"], 75.0)
        # 4 out of 4 (100%) are within 2500km
        self.assertEqual(metrics["Acc@2500km"], 100.0)

    def test_mismatch_length_raises(self):
        targets = [(10.795, 106.721)]
        predictions = [(10.795, 106.721), (21.028, 105.834)]

        with self.assertRaises(ValueError):
            calculate_batch_haversine_distance(targets, predictions)

        with self.assertRaises(ValueError):
            compute_distance_accuracy_metrics(targets, predictions)


if __name__ == "__main__":
    unittest.main()
