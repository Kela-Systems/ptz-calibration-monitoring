#!/usr/bin/env python3
"""
Test script for the calibration monitoring orchestration.

This script tests the orchestrator's functionality without requiring
actual camera hardware or AWS credentials.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import run_calibration_monitoring module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_calibration_monitoring",
    str(Path(__file__).parent / "run_calibration_monitoring.py")
)
run_calibration_monitoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_calibration_monitoring)
CalibrationMonitoringOrchestrator = run_calibration_monitoring.CalibrationMonitoringOrchestrator


class TestCalibrationOrchestrator(unittest.TestCase):
    """Test cases for the CalibrationMonitoringOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test devices.yaml
        self.devices_yaml = Path(self.temp_dir) / "devices.yaml"
        devices_data = [
            {
                "name": "test-deployment-1",
                "cameras": [
                    {"name": "camera-1"},
                    {"name": "camera-2"}
                ]
            },
            {
                "name": "test-deployment-2",
                "cameras": [
                    {"name": "camera-3"}
                ]
            }
        ]
        with open(self.devices_yaml, 'w') as f:
            yaml.dump(devices_data, f)
        
        # Create test camera calibration
        self.camera_intrinsics = Path(self.temp_dir) / "calibration.npz"
        camera_matrix = np.eye(3)
        dist_coeff = np.zeros((1, 5))
        np.savez(self.camera_intrinsics, camera_matrix=camera_matrix, dist_coeff=dist_coeff)
        
        # Create test R_align
        self.r_align_path = Path(self.temp_dir) / "r_align.npy"
        r_align = np.eye(3)
        np.save(self.r_align_path, r_align)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('monitoring.calibration_runner.CalibrationRunner')
    @patch('monitoring.aws_integration.AWSIntegration')
    @patch('monitoring.slack_notifier.SlackNotifier')
    def test_orchestrator_initialization(self, mock_slack, mock_aws, mock_calibration):
        """Test orchestrator initialization."""
        orchestrator = CalibrationMonitoringOrchestrator(
            devices_yaml_path=str(self.devices_yaml),
            camera_intrinsics_path=str(self.camera_intrinsics),
            r_align_path=str(self.r_align_path)
        )
        
        # Check devices were loaded
        self.assertEqual(len(orchestrator.devices), 3)
        self.assertEqual(orchestrator.devices[0]['deployment_name'], 'test-deployment-1')
        self.assertEqual(orchestrator.devices[0]['device_id'], 'camera-1')
        self.assertEqual(orchestrator.devices[1]['device_id'], 'camera-2')
        self.assertEqual(orchestrator.devices[2]['deployment_name'], 'test-deployment-2')
        self.assertEqual(orchestrator.devices[2]['device_id'], 'camera-3')
    
    @patch('subprocess.run')
    def test_kubectl_context_switch(self, mock_subprocess):
        """Test kubectl context switching."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('monitoring.calibration_runner.CalibrationRunner'), \
             patch('monitoring.aws_integration.AWSIntegration'), \
             patch('monitoring.slack_notifier.SlackNotifier'):
            
            orchestrator = CalibrationMonitoringOrchestrator(
                devices_yaml_path=str(self.devices_yaml),
                camera_intrinsics_path=str(self.camera_intrinsics),
                r_align_path=str(self.r_align_path)
            )
            
            result = orchestrator._switch_kubectl_context('test-deployment-1')
            self.assertTrue(result)
            
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            self.assertEqual(call_args[0], 'kubectl')
            self.assertEqual(call_args[1], 'config')
            self.assertEqual(call_args[2], 'use-context')
            self.assertEqual(call_args[3], 'test-deployment-1')
    
    @patch('monitoring.query_extractor.extract_query')
    def test_extract_query_frames(self, mock_extract):
        """Test query frame extraction."""
        # Mock successful extraction
        mock_extract.return_value = {
            'camera_frames': [Mock(), Mock()],
            'telemetry': {0: {}, 1: {}},
            'capture_positions': [(0.0, 0.0, 1.0), (45.0, -10.0, 1.0)],
            'temp_dir': '/tmp/test',
            'manifest_path': '/tmp/test/manifest.json'
        }
        
        with patch('monitoring.calibration_runner.CalibrationRunner'), \
             patch('monitoring.aws_integration.AWSIntegration'), \
             patch('monitoring.slack_notifier.SlackNotifier'):
            
            orchestrator = CalibrationMonitoringOrchestrator(
                devices_yaml_path=str(self.devices_yaml),
                camera_intrinsics_path=str(self.camera_intrinsics),
                r_align_path=str(self.r_align_path)
            )
            
            result = orchestrator._extract_query_frames('camera-1')
            
            self.assertIsNotNone(result)
            self.assertEqual(len(result['camera_frames']), 2)
            mock_extract.assert_called_once_with(
                device_id='camera-1',
                stabilize_time=30,
                timeout=None,
                active_ptz_stops=None
            )
    
    def test_load_devices_error_handling(self):
        """Test error handling when devices.yaml is invalid."""
        invalid_yaml = Path(self.temp_dir) / "invalid.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [[[")
        
        with patch('monitoring.calibration_runner.CalibrationRunner'), \
             patch('monitoring.aws_integration.AWSIntegration'), \
             patch('monitoring.slack_notifier.SlackNotifier'):
            
            orchestrator = CalibrationMonitoringOrchestrator(
                devices_yaml_path=str(invalid_yaml),
                camera_intrinsics_path=str(self.camera_intrinsics),
                r_align_path=str(self.r_align_path)
            )
            
            # Should return empty list on error
            self.assertEqual(len(orchestrator.devices), 0)
    
    @patch('monitoring.query_extractor.extract_query')
    @patch('subprocess.run')
    def test_process_device_success(self, mock_subprocess, mock_extract):
        """Test successful device processing."""
        # Mock kubectl context switch
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Mock query extraction
        temp_test_dir = tempfile.mkdtemp()
        mock_extract.return_value = {
            'camera_frames': [Mock()],
            'telemetry': {0: {}},
            'capture_positions': [(0.0, 0.0, 1.0)],
            'temp_dir': temp_test_dir,
            'manifest_path': f'{temp_test_dir}/manifest.json'
        }
        
        with patch('monitoring.calibration_runner.CalibrationRunner') as mock_calibration_class, \
             patch('monitoring.aws_integration.AWSIntegration') as mock_aws_class, \
             patch('monitoring.slack_notifier.SlackNotifier') as mock_slack_class:
            
            # Mock calibration result
            mock_result = Mock()
            mock_result.median_pitch = 0.2
            mock_result.median_yaw = 0.3
            mock_result.median_roll = 0.1
            
            mock_calibration_instance = Mock()
            mock_calibration_instance.run_full_calibration_pipeline.return_value = mock_result
            mock_calibration_class.return_value = mock_calibration_instance
            
            # Mock AWS integration
            mock_aws_instance = Mock()
            mock_aws_instance.upload_directory.return_value = ['s3://test/file1.png']
            mock_aws_instance.get_s3_path.return_value = 's3://test/path/'
            mock_aws_instance.write_calibration_result.return_value = 'query-123'
            mock_aws_class.return_value = mock_aws_instance
            
            # Mock Slack notifier
            mock_slack_instance = Mock()
            mock_slack_instance.send_calibration_report.return_value = True
            mock_slack_class.return_value = mock_slack_instance
            
            orchestrator = CalibrationMonitoringOrchestrator(
                devices_yaml_path=str(self.devices_yaml),
                camera_intrinsics_path=str(self.camera_intrinsics),
                r_align_path=str(self.r_align_path)
            )
            
            success = orchestrator.process_device(
                deployment_name='test-deployment-1',
                device_id='camera-1'
            )
            
            self.assertTrue(success)
            mock_calibration_instance.run_full_calibration_pipeline.assert_called_once()
            mock_aws_instance.write_calibration_result.assert_called_once()
            mock_slack_instance.send_calibration_report.assert_called_once()


def main():
    """Run the tests."""
    # Run with verbose output
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()

