import os
import cv2
import json
from .utils import test_homography_roundtrip, estimate_camera_intrinsics, normalize_angle_delta

def main():
    PATH_TO_REFERENCE_FOLDER = 'C:/Users/vasil/Documents/data/PTZ_Calibration_and_georegistration/capture_output_56/capture_output_56/frames'
    PATH_TO_REFERENCE_MANIFEST = os.path.join(PATH_TO_REFERENCE_FOLDER, 'manifest.json')
    REFERENCE_PANORAMA_PATH = './data/referencePanorama.json'

    PATH_TO_QUERY_IMAGES = 'C:/Users/vasil/Documents/data/PTZ_Calibration_and_georegistration/capture_output/capture_output/frames'
    PATH_TO_QUERY_MANIFEST = os.path.join(PATH_TO_QUERY_IMAGES, 'manifest.json')

    query_file = 'frame_0001_yaw_60.0_pitch_-88.0.png'
    ref_file = 'frame_0002_yaw_80.0_pitch_-88.0.png'

    query_path = os.path.join(PATH_TO_QUERY_IMAGES, query_file)
    ref_path = os.path.join(PATH_TO_REFERENCE_FOLDER, ref_file)

    query_image = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    ref_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    # Load the query manifest file
    try:
        with open(PATH_TO_QUERY_MANIFEST, 'r') as f:
            query_manifest = json.load(f)
    except FileNotFoundError:
        print(f"Error: Query manifest file not found at {PATH_TO_QUERY_MANIFEST}")
        return []
    
    try:
        with open(PATH_TO_REFERENCE_MANIFEST, 'r') as f:
            ref_manifest = json.load(f)
    except FileNotFoundError:
        print(f"Error: Query manifest file not found at {PATH_TO_REFERENCE_MANIFEST}")
        return []
    
    query_image_height, query_image_width = query_image.shape[:2]
    query_manifest_filename = query_manifest[query_file]
    fov_v = query_manifest_filename['field_of_view_v']
    fov_h = query_manifest_filename['field_of_view_h']
    K_query = estimate_camera_intrinsics(query_image_width, query_image_height, fov_v, fov_h)

    ref_manifest_filename = ref_manifest[ref_file]
    ref_image_height, ref_image_width = ref_image.shape[:2]
    fov_v = ref_manifest_filename['field_of_view_v']
    fov_h = ref_manifest_filename['field_of_view_h']
    K_ref = estimate_camera_intrinsics(ref_image_width, ref_image_height, fov_v, fov_h)

    query_telemetry = query_manifest_filename['attitude']
    ref_telemetry = ref_manifest_filename['attitude']

    telemetry_offset_yaw = query_telemetry['yaw'] - ref_telemetry['yaw']
    telemetry_offset_yaw = normalize_angle_delta(telemetry_offset_yaw)
    telemetry_offset_pitch = query_telemetry['pitch'] - ref_telemetry['pitch']
    telemetry_offset_pitch = normalize_angle_delta(telemetry_offset_pitch)
    telemetry_offset_roll = query_telemetry['roll'] - ref_telemetry['roll']
    telemetry_offset_roll = normalize_angle_delta(telemetry_offset_roll)


    telemetry_offset = {
                'yaw': telemetry_offset_yaw,
                'pitch': telemetry_offset_pitch,
                'roll': telemetry_offset_roll
                }

    test_homography_roundtrip(query_image, ref_image, K_query, K_ref, telemetry_offset)
    pass

if __name__ == '__main__':
    main()