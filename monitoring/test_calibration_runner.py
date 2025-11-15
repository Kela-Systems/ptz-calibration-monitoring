#!/usr/bin/env python3
"""
Unit tests for CalibrationRunner module.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import torch

from calibration_runner import (
    CalibrationRunner,
    CalibrationConfig,
    CalibrationResult,
    load_camera_calibration,
    load_r_align
)


class TestCalibrationConfig(unittest.TestCase):
    """Test CalibrationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CalibrationConfig()
        
        self.assertEqual(config.min_match_count, 15)
        self.assertEqual(config.ransac_threshold, 5.0)
        self.assertEqual(config.roi, [0.1, 0.1, 0.9, 0.9])
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CalibrationConfig(
            min_match_count=20,
            ransac_threshold=3.0,
            roi=[0.0, 0.0, 1.0, 1.0]
        )
        
        self.assertEqual(config.min_match_count, 20)
        self.assertEqual(config.ransac_threshold, 3.0)
        self.assertEqual(config.roi, [0.0, 0.0, 1.0, 1.0])


class TestCalibrationResult(unittest.TestCase):
    """Test CalibrationResult dataclass."""
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = CalibrationResult(
            median_yaw=1.0,
            median_pitch=2.0,
            median_roll=3.0,
            weighted_mean_yaw=1.1,
            weighted_mean_pitch=2.1,
            weighted_mean_roll=3.1,
            std_yaw=0.1,
            std_pitch=0.2,
            std_roll=0.3,
            mean_angular_distance=0.5,
            num_high_confidence_matches=10,
            all_offsets=[]
        )
        
        result_dict = result.to_dict()
        
        self.assertIn('median_offsets', result_dict)
        self.assertIn('weighted_mean_offsets', result_dict)
        self.assertIn('standard_deviations', result_dict)
        self.assertIn('confidence_metrics', result_dict)
        
        self.assertEqual(result_dict['median_offsets']['yaw'], 1.0)
        self.assertEqual(result_dict['confidence_metrics']['num_high_confidence_matches'], 10)


class TestCalibrationRunner(unittest.TestCase):
    """Test CalibrationRunner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CalibrationRunner(
            s3_bucket="test-bucket",
            aws_region="us-east-1"
        )
    
    @patch('calibration_runner.boto3.client')
    def test_initialization(self, mock_boto_client):
        """Test CalibrationRunner initialization."""
        runner = CalibrationRunner(
            s3_bucket="test-bucket",
            aws_region="us-west-2"
        )
        
        self.assertEqual(runner.s3_bucket, "test-bucket")
        self.assertEqual(runner.aws_region, "us-west-2")
        self.assertIsNotNone(runner.config)
    
    @patch('calibration_runner.boto3.client')
    def test_download_s3_directory_success(self, mock_boto_client):
        """Test successful S3 directory download."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        
        # Mock paginator
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {'Key': 'prefix/file1.txt'},
                    {'Key': 'prefix/file2.txt'}
                ]
            }
        ]
        
        runner = CalibrationRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner._download_s3_directory("prefix/", temp_dir)
            
            self.assertTrue(result)
            self.assertEqual(mock_s3.download_file.call_count, 2)
    
    def test_load_reference_map_from_colmap(self):
        """Test loading reference map from COLMAP files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock reference_panorama.json
            reference_data = {
                "frames": [
                    {
                        "filename": "frame_001.png",
                        "attitude": {
                            "yaw": 0.0,
                            "pitch": 0.0,
                            "roll": 0.0
                        }
                    }
                ]
            }
            
            reference_json_path = os.path.join(temp_dir, "reference_panorama.json")
            with open(reference_json_path, 'w') as f:
                json.dump(reference_data, f)
            
            # Create mock COLMAP feature file
            feature_file_path = os.path.join(temp_dir, "frame_001.txt")
            with open(feature_file_path, 'w') as f:
                # COLMAP format: num_keypoints descriptor_dim
                # Then: x y scale orientation descriptor_values...
                f.write("2 128\n")
                f.write("100.0 200.0 1.0 0.0 " + " ".join(["0.5"] * 128) + "\n")
                f.write("150.0 250.0 1.0 0.0 " + " ".join(["0.5"] * 128) + "\n")
            
            # Load reference map
            reference_map = self.runner._load_reference_map_from_colmap(
                temp_dir,
                reference_json_path
            )
            
            self.assertIsNotNone(reference_map)
            self.assertEqual(len(reference_map), 1)
            self.assertEqual(reference_map[0]['filename'], 'frame_001.png')
            self.assertIn('keypoints', reference_map[0])
            self.assertIn('descriptors', reference_map[0])
            self.assertIn('telemetry', reference_map[0])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_load_camera_calibration(self):
        """Test loading camera calibration from .npz file."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Create mock calibration data
            camera_matrix = np.eye(3)
            dist_coeff = np.zeros((5, 1))
            
            np.savez(temp_path, camera_matrix=camera_matrix, dist_coeff=dist_coeff)
            
            try:
                # Load calibration
                loaded_matrix, loaded_coeff = load_camera_calibration(temp_path)
                
                np.testing.assert_array_equal(loaded_matrix, camera_matrix)
                np.testing.assert_array_equal(loaded_coeff, dist_coeff)
            finally:
                os.unlink(temp_path)
    
    def test_load_r_align(self):
        """Test loading R_align matrix."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Create mock R_align matrix
            r_align = np.eye(3)
            np.save(temp_path, r_align)
            
            try:
                # Load R_align
                loaded_r_align = load_r_align(temp_path)
                
                np.testing.assert_array_equal(loaded_r_align, r_align)
            finally:
                os.unlink(temp_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for calibration pipeline."""
    
    @patch('calibration_runner.calculate_offsets_from_visual_matches')
    @patch('calibration_runner.calculate_final_offset')
    def test_run_calibration_success(self, mock_calc_final, mock_calc_offsets):
        """Test successful calibration run."""
        # Mock return values
        mock_offsets = [
            {
                'euler_angle_offset_yaw': 1.0,
                'euler_angle_offset_pitch': 2.0,
                'euler_angle_offset_roll': 3.0,
                'angular distance': 0.5,
                'number_of_good_matches': 50
            }
        ]
        mock_calc_offsets.return_value = mock_offsets
        
        mock_final = {
            'median_yaw': 1.0,
            'median_pitch': 2.0,
            'median_roll': 3.0,
            'weighted_mean_yaw': 1.1,
            'weighted_mean_pitch': 2.1,
            'weighted_mean_roll': 3.1,
            'std_yaw_offset': 0.1,
            'std_pitch_offset': 0.2,
            'std_roll_offset': 0.3,
            'mean_angular_distances': 0.5
        }
        mock_calc_final.return_value = mock_final
        
        # Set up test data
        runner = CalibrationRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock manifest
            manifest_path = os.path.join(temp_dir, "manifest.json")
            manifest_data = {
                "frame_001.png": {
                    "attitude": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
                }
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
            
            # Create mock query image
            query_folder = os.path.join(temp_dir, "query")
            os.makedirs(query_folder)
            # Note: In real test, we'd create an actual image file
            
            # Run calibration
            reference_map = []
            camera_matrix = np.eye(3)
            dist_coeff = np.zeros((5, 1))
            r_align = np.eye(3)
            
            result = runner.run_calibration(
                query_folder=query_folder,
                query_manifest_path=manifest_path,
                reference_map=reference_map,
                camera_matrix=camera_matrix,
                dist_coeff=dist_coeff,
                r_align=r_align
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, CalibrationResult)
            self.assertEqual(result.median_yaw, 1.0)
            self.assertEqual(result.num_high_confidence_matches, 1)


if __name__ == '__main__':
    unittest.main()

