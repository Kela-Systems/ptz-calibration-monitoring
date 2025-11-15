#!/usr/bin/env python3
"""
Simple test script to validate the ReferenceCollector module without running full capture.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from monitoring import ReferenceCollector
        print("✓ ReferenceCollector imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ReferenceCollector: {e}")
        return False

def test_initialization():
    """Test that ReferenceCollector can be initialized."""
    print("\nTesting initialization...")
    try:
        from monitoring import ReferenceCollector
        
        collector = ReferenceCollector(
            devices_yaml_path="devices.yaml",
            s3_bucket="camera-calibration-monitoring",
            aws_region="us-east-1"
        )
        
        print(f"✓ ReferenceCollector initialized successfully")
        print(f"  - Loaded {len(collector.devices)} device(s)")
        
        # Display devices
        for device in collector.devices:
            device_name = device.get("name", "Unknown")
            cameras = device.get("cameras", [])
            print(f"  - Device: {device_name}")
            for camera in cameras:
                camera_name = camera.get("name", "Unknown")
                print(f"    - Camera: {camera_name}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize ReferenceCollector: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_port_forward_utils():
    """Test that port_forward_utils can be imported."""
    print("\nTesting port_forward_utils...")
    try:
        import port_forward_utils
        print("✓ port_forward_utils imported successfully")
        print(f"  - start_port_forwards: {hasattr(port_forward_utils, 'start_port_forwards')}")
        print(f"  - stop_port_forwards: {hasattr(port_forward_utils, 'stop_port_forwards')}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import port_forward_utils: {e}")
        return False

def test_devices_yaml():
    """Test that devices.yaml can be parsed."""
    print("\nTesting devices.yaml parsing...")
    try:
        import yaml
        with open("devices.yaml", 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"✓ devices.yaml parsed successfully")
        print(f"  - Found {len(data)} device(s)")
        
        for device in data:
            device_name = device.get("name", "Unknown")
            cameras = device.get("cameras", [])
            print(f"  - {device_name}: {len(cameras)} camera(s)")
        
        return True
    except Exception as e:
        print(f"✗ Failed to parse devices.yaml: {e}")
        return False

def test_collector_methods():
    """Test that ReferenceCollector has expected methods."""
    print("\nTesting ReferenceCollector methods...")
    try:
        from monitoring import ReferenceCollector
        
        expected_methods = [
            'collect_reference_for_device',
            'collect_all_references',
            '_switch_kubectl_context',
            '_capture_reference_scan',
            '_extract_features',
            '_upload_to_s3'
        ]
        
        all_present = True
        for method_name in expected_methods:
            has_method = hasattr(ReferenceCollector, method_name)
            status = "✓" if has_method else "✗"
            print(f"  {status} {method_name}")
            if not has_method:
                all_present = False
        
        return all_present
    except Exception as e:
        print(f"✗ Failed to test methods: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Reference Collector Module Validation")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("Port Forward Utils Test", test_port_forward_utils),
        ("Devices YAML Test", test_devices_yaml),
        ("Methods Test", test_collector_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status:6s} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Module is ready for testing with actual devices.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

