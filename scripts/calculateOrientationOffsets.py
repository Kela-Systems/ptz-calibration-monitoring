import os
import configparser
import argparse
import numpy as np
import pandas as pd
from ptz_georeg.utils import build_reference_map, save_reference_map, load_reference_map, AppConfig, calculate_offsets_from_visual_matches, calculate_final_offset
from ptz_georeg.utils import MatchingMethod, visualize_matches_custom, visualize_transformations, GeometryModel, visualize_epilines
from ptz_georeg.utils import estimate_camera_intrinsics, run_ransac_for_alignment, evaluate_R_align, rescale_intrinsic_camera_properties
from ptz_georeg.utils import calculate_new_dimensions, resize_images_in_folder, recreate_directory
from typing import List
from ptz_georeg.match_info import SensorTelemetryPair

def main(config:AppConfig):
    calib_data = np.load(config.camera_intrinsics_path)
    camMatrix = calib_data['camMatrix']
    distCoeff = calib_data['distCoeff']
    if(config.align_rotation_matrix):
        r_align = np.load(config.r_align_path)
    else:
        r_align = np.eye(3)
    
    if(config.to_estimate_camera_intrinsics):
        dict_estimate_params = config.estimate_camera_intrinsics_params
        image_width = dict_estimate_params['image_width']
        image_height = dict_estimate_params['image_height']
        field_of_view_v = dict_estimate_params['field_of_view_v']
        field_of_view_h = dict_estimate_params['field_of_view_h']
        camMatrix = estimate_camera_intrinsics(image_width, image_height, field_of_view_v, field_of_view_h)
        distCoeff = np.array([[0., 0., 0., 0., 0.]])
    else:
        new_width = estimate_camera_intrinsics_params['image_width']
        new_height = estimate_camera_intrinsics_params['image_height']
        resize_height, resize_width = calculate_new_dimensions(new_height, new_width, config.max_dimension)
        camMatrix = rescale_intrinsic_camera_properties(camMatrix, 
                                                        config.width_intrinsics_path,
                                                        config.height_intrinsics_path, 
                                                        resize_width, 
                                                        resize_height)
        pass
    
    resize_images_in_folder(config.reference_image_folder, config.max_dimension, config.reference_image_folder_temp)
    resize_images_in_folder(config.query_image_folder, config.max_dimension, config.query_image_folder_temp)

    reference_panorama = build_reference_map(config, camMatrix, distCoeff)
    save_reference_map(reference_panorama, config.map_save_path)
    reference_panorama = load_reference_map(config.map_save_path, config.matching_method)
    if reference_panorama is None:
        print("Reference map not found. Please run the script to create it first.")
    else:
        visual_telemetry_offsets = calculate_offsets_from_visual_matches(
                config.query_image_folder_temp,
                config.query_manifest_path,
                reference_panorama,
                config.min_match_count,
                config.matching_method,
                config.ransac_threshold,
                camMatrix,
                distCoeff,
                config.geometry_model,
                r_align,
                config.roi,
                config.num_features_deep_learning,
                config.resize_scale_deep_learning,
                config.max_xy_norm,
                config.min_overlapping_ratio
            )
        
        nr_of_visual_telemetry_offsets = len(visual_telemetry_offsets)
        # --- Do Ransac Algorithm to Keep Best inlier offsets ---
        print("")
        print("\n--- Ransac for Outlier Offset Rejection ---")
        print("")
        visual_telemetry_offsets_inliers = []
        if visual_telemetry_offsets:
            sensorTelemetryPairs:List[SensorTelemetryPair] = []
            for offset in visual_telemetry_offsets:
                sensorTelemetryPair = SensorTelemetryPair()
                sensorTelemetryPair.sensor_rotation_matrix = offset['physical_rotation_matrix']
                sensorTelemetryPair.telemetry_rotation_matrix = offset['telemetry_rotation_matrix']
                sensorTelemetryPairs.append(sensorTelemetryPair)
            
            R_align_final, best_model_inliers, best_model_inliers_indices = run_ransac_for_alignment(sensorTelemetryPairs, n=4, k=100, t_deg=config.ransac_angular_distance_threshold)
            print("evaluate R align function with Ransac, r_align_final: ")
            evaluate_R_align(R_align_final, sensorTelemetryPairs)

            print("")
            print("evaluate R align function with Ransac on inlier pairs, r_align_final: ")
            evaluate_R_align(R_align_final, best_model_inliers)
            print("")
            print("Final Align Matrix: ", R_align_final)

            print("All pairs: ", len(sensorTelemetryPairs))
            print("Inlier pairs: ", len(best_model_inliers))
            for index in best_model_inliers_indices:
                visual_telemetry_offsets_inliers.append(visual_telemetry_offsets[index])

        nr_of_visual_telemetry_offsets_inliers = len(visual_telemetry_offsets_inliers)
        visual_telemetry_offsets = visual_telemetry_offsets_inliers
        # --- Display the results ---
        print("\n--- Calculated Individual Offsets ---")
        if visual_telemetry_offsets:
            for offset in visual_telemetry_offsets:
                print("")
                with np.printoptions(precision=4, suppress=True):
                    print(
                    f"""
                    --------------------------------------------------
                    Query Frame:         {offset['query_filename']}
                    Best Match Ref:      {offset['best_match_ref_filename']}
                    Number of Matches:   {offset['number_of_good_matches']}

                    --- Telemetry Offsets Between Matched Frames ---
                    Yaw:                 {offset['telemetry_offset_yaw']:.4f}
                    Pitch:               {offset['telemetry_offset_pitch']:.4f}
                    Roll:                {offset['telemetry_offset_roll']:.4f}

                    --- Offsets Between Physical and Telemetry Rotation Matrices ---
                    Yaw:                 {offset['physical_telemetry_offset_yaw']:.4f}
                    Pitch:               {offset['physical_telemetry_offset_pitch']:.4f}
                    Roll:                {offset['physical_telemetry_offset_roll']:.4f}

                    --- Offsets Between Physical and Telemetry Query Euler Angles Difference ---
                    Yaw:                 {offset['euler_angle_offset_yaw']:.4f}
                    Pitch:               {offset['euler_angle_offset_pitch']:.4f}
                    Roll:                {offset['euler_angle_offset_roll']:.4f}

                    --- Rotation Matrix (from Matching Points) ---
                    R_Physical: 
                    {offset['physical_rotation_matrix']}

                    --- Rotation Matrix (from Telemetry Data) ---
                    R_telemetry: 
                    {offset['telemetry_rotation_matrix']}
                    --------------------------------------------------

                    --- Their Angular Distance is ---
                    Angular distance:    {offset['angular distance']:.4f}

                    --- The euler sequence to build telemetry matrix --
                    Euler Sequence: {offset['euler sequence']}

                    """
                    )
                print("")


            data_for_excel = []
            for offset in visual_telemetry_offsets:
                row_data = {
                    'yaw_offset': offset['euler_angle_offset_yaw'],
                    'pitch_offset': offset['euler_angle_offset_pitch'],
                    'roll_offset': offset['euler_angle_offset_roll'],
                    'query_name': offset['query_filename'],
                    'ref_name': offset['best_match_ref_filename'],
                    'xy_norm': offset['xy norm'],
                    'num_matches': offset['number_of_good_matches'],
                    'overlapping ratio': offset['overlapping ratio']
                }
                data_for_excel.append(row_data)
            df = pd.DataFrame(data_for_excel)
            df.to_excel(config.output_excel_path, index=False)
            print(f"Successfully saved data to {config.output_excel_path}")

            final_offset_results = calculate_final_offset(visual_telemetry_offsets)
            if final_offset_results:
                print("\n--- Final System Offset Estimation ---")
                print(f"Number of high-confidence matches: {nr_of_visual_telemetry_offsets}")
                print(f"Number of high-confidence samples used after Ransac Algorithm: {final_offset_results['number_of_inlier_frames']}")
                
                # --- The final calculated offset ---
                median_offset = final_offset_results['median_offset']
                weighted_mean_offset = final_offset_results['weighted_mean_offset']
                
                mean_angular_distance = final_offset_results["mean angular distance"]
                
                print("\n--- Calculated Global Offset ---")
                print(f"Median Offset (Yaw, Pitch, Roll):        ({median_offset['yaw']:8.4f}, {median_offset['pitch']:8.4f}, {median_offset['roll']:8.4f})")
                print(f"Weighted Mean Offset (Yaw, Pitch, Roll): ({weighted_mean_offset['yaw']:8.4f}, {weighted_mean_offset['pitch']:8.4f}, {weighted_mean_offset['roll']:8.4f})")

                std_dev = final_offset_results['std']        
                print("\n--- Consistency of Measurements (Standard Deviation) ---")
                print(f"Std Dev of Offsets (Yaw, Pitch, Roll):              ({std_dev['yaw']:8.4f}, {std_dev['pitch']:8.4f}, {std_dev['roll']:8.4f})")
                print(f"Mean angular distance: {mean_angular_distance:8.4f}")
                # ----------------------------
            else:
                print("No valid offsets could be calculated.")

            
            if(config.visualize_matches):
                recreate_directory(config.save_matching_imgs_path)
                recreate_directory(config.transformed_images_path)
                for offset in visual_telemetry_offsets:
                    query_path = os.path.join(config.query_image_folder_temp, offset['query_filename'])
                    ref_path = os.path.join(config.reference_image_folder_temp, offset['best_match_ref_filename'])
                    output_filename = offset['query_filename'] + offset['best_match_ref_filename']
                    output_path = os.path.join(config.save_matching_imgs_path, output_filename)
                    visualize_matches_custom(query_path, 
                                             ref_path, 
                                             offset['matched_keypoints_query'], 
                                             offset['matched_keypoints_ref'],
                                             camMatrix,
                                             distCoeff,
                                             output_path)
                    save_path_side_name = offset['query_filename'] + offset['best_match_ref_filename'] + 'side.png'
                    save_path_overlay_name = offset['query_filename'] + offset['best_match_ref_filename'] + 'overlay.png'
                    save_path_side = os.path.join(config.transformed_images_path, save_path_side_name)
                    save_path_overlay = os.path.join(config.transformed_images_path, save_path_overlay_name)
                    if(config.geometry_model==GeometryModel.HOMOGRAPHY):
                        visualize_transformations(offset['homography_matrix'], 
                                              query_path, 
                                              ref_path, 
                                              save_path_side, 
                                              save_path_overlay,
                                              camMatrix,
                                              distCoeff)
                    if(config.geometry_model==GeometryModel.ESSENTIAL):
                        visualize_epilines(
                        query_path,          # Path to the original query image
                        ref_path,            # Path to the original reference image
                        offset['matched_keypoints_query'],       # The filtered inlier points
                        offset['matched_keypoints_ref'],         # The filtered inlier points
                        offset['homography_matrix'],                   # The calculated Essential Matrix
                        camMatrix,        # Your calibrated K matrix
                        save_path_side # The output path
                    )
          
if __name__ == '__main__':
    # --- 1. SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Calculate orientation offsets between query and reference images.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration INI file.')
    args = parser.parse_args()

    # --- 2. READ THE CONFIGURATION FILE ---
    config_parser = configparser.ConfigParser()
    config_parser.read(args.config)

    # --- 3. EXTRACT VALUES FROM CONFIG FILE ---
    # Paths section
    paths = config_parser['paths']
    PATH_TO_REFERENCE_FOLDER = paths.get('reference_folder')
    PATH_TO_QUERY_IMAGES = paths.get('query_images_folder')
    r_align_path = paths.get('r_align_path')
    output_filename = paths.get('output_excel_path')
    PATH_TO_REFERENCE_MANIFEST = os.path.join(PATH_TO_REFERENCE_FOLDER, 'manifest.json')
    PATH_TO_QUERY_MANIFEST = os.path.join(PATH_TO_QUERY_IMAGES, 'manifest.json')
    REFERENCE_PANORAMA_PATH = paths.get('reference_panorama_path')
    camera_intrinsics_path = paths.get('camera_intrinsics_path')
    save_matching_imgs_path = paths.get('save_matching_imgs_path')
    transformed_images_path = paths.get('transformed_images_path')
    reference_image_folder_temp = paths.get('reference_image_folder_temp')
    query_image_folder_temp = paths.get('query_image_folder_temp')

    # Parameters section
    params = config_parser['parameters']
    number_minimum_matches = params.getint('min_match_count')
    ransac_threshold = params.getfloat('ransac_threshold')
    geometry_model = GeometryModel[params.get('geometry_model').upper()]
    feature_matching_method = MatchingMethod[params.get('feature_matching_method').upper()]
    align_rotation_matrix = params.getboolean('align_rotation_matrix')
    visualize_matches = params.getboolean('visualize_matches')

    # Image Processing section
    img_proc = config_parser['image_processing']
    roi_str = img_proc.get('roi_crop_percentages')
    ROI = [float(x.strip()) for x in roi_str.split(',')]
    max_dimension = img_proc.getint('max_dimension')

    # Deep Learning section
    dl = config_parser['deep_learning']
    num_features_deep_learning = dl.getint('num_features')
    resize_scale_deep_learning = dl.getfloat('resize_scale')

    # Camera Intrinsics section
    cam_intr = config_parser['camera_intrinsics']
    to_estimate_camera_intrinsics = cam_intr.getboolean('to_estimate')
    width_camera_intrinsics = cam_intr.getint('width_intrinsics_path')
    height_camera_intrinsics = cam_intr.getint('height_intrinsics_path')

    # Intrinsics Estimation section
    est_intr = config_parser['intrinsics_estimation']
    estimate_camera_intrinsics_params = {
        'image_width': est_intr.getint('image_width'),
        'image_height': est_intr.getint('image_height'),
        'field_of_view_v': est_intr.getfloat('field_of_view_v'),
        'field_of_view_h': est_intr.getfloat('field_of_view_h')
    }

    # Analysis Tuning section
    tuning = config_parser['analysis_tuning']
    ransac_angular_distance_threshold = tuning.getfloat('ransac_angular_distance_threshold')
    max_xy_norm = tuning.getfloat('max_xy_norm')
    min_overlapping_ratio = tuning.getfloat('min_overlapping_ratio')

    # --- 4. POPULATE THE AppConfig OBJECT ---
    config = AppConfig(
        reference_image_folder=PATH_TO_REFERENCE_FOLDER,
        reference_manifest_path=PATH_TO_REFERENCE_MANIFEST,
        reference_image_folder_temp=reference_image_folder_temp,
        query_image_folder_temp=query_image_folder_temp,
        map_save_path=REFERENCE_PANORAMA_PATH,
        query_image_folder=PATH_TO_QUERY_IMAGES,
        query_manifest_path=PATH_TO_QUERY_MANIFEST,
        min_match_count=number_minimum_matches,
        matching_method=feature_matching_method,
        ransac_threshold=ransac_threshold,
        visualize_matches=visualize_matches,
        save_matching_imgs_path=save_matching_imgs_path,
        transformed_images_path=transformed_images_path,
        camera_intrinsics_path=camera_intrinsics_path,
        r_align_path=r_align_path,
        align_rotation_matrix=align_rotation_matrix,
        geometry_model=geometry_model,
        to_estimate_camera_intrinsics=to_estimate_camera_intrinsics,
        estimate_camera_intrinsics_params=estimate_camera_intrinsics_params,
        roi=ROI,
        num_features_deep_learning=num_features_deep_learning,
        resize_scale_deep_learning=resize_scale_deep_learning,
        output_excel_path=output_filename,
        ransac_angular_distance_threshold=ransac_angular_distance_threshold,
        max_dimension=max_dimension,
        max_xy_norm=max_xy_norm,
        min_overlapping_ratio=min_overlapping_ratio,
        width_intrinsics_path=width_camera_intrinsics,
        height_intrinsics_path=height_camera_intrinsics
    )

    # --- 5. RUN THE MAIN FUNCTION ---
    main(config)