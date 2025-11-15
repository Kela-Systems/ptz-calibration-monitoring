import os
import numpy as  np
import configparser
import argparse
from ptz_georeg.utils import MatchingMethod, GeometryModel, AppConfig
from ptz_georeg.utils import build_reference_map, save_reference_map, load_reference_map, resize_images_in_folder
from ptz_georeg.utils import calculate_R_align_from_reference_frames, find_R_align, run_ransac_for_alignment
from ptz_georeg.utils import evaluate_R_align, estimate_camera_intrinsics, calculate_R_align_from_reference_frames_lightglue, rescale_intrinsic_camera_properties
from ptz_georeg.utils import calculate_new_dimensions, print_matching_statistics_summary

def main(config:AppConfig, save_path:str):
    if(config.to_estimate_camera_intrinsics):
        dict_estimate_params = config.estimate_camera_intrinsics_params
        image_width = dict_estimate_params['image_width']
        image_height = dict_estimate_params['image_height']
        field_of_view_v = dict_estimate_params['field_of_view_v']
        field_of_view_h = dict_estimate_params['field_of_view_h']
        camMatrix = estimate_camera_intrinsics(image_width, image_height, field_of_view_v, field_of_view_h)
        distCoeff = np.array([[0., 0., 0., 0., 0.]])
    else:
        calib_data = np.load(config.camera_intrinsics_path)
        camMatrix = calib_data['camMatrix']
        distCoeff = calib_data['distCoeff']
        new_width = config.estimate_camera_intrinsics_params['image_width']
        new_height = config.estimate_camera_intrinsics_params['image_height']
        resize_height, resize_width = calculate_new_dimensions(new_height, new_width, config.max_dimension)
        camMatrix = rescale_intrinsic_camera_properties(camMatrix, 
                                                        config.width_intrinsics_path, 
                                                        config.height_intrinsics_path, 
                                                        resize_width, 
                                                        resize_height)
        

    resize_images_in_folder(config.reference_image_folder, 
                            config.max_dimension, 
                            config.reference_image_folder_temp)


    reference_panorama = build_reference_map(config, camMatrix, distCoeff)
    save_reference_map(reference_panorama, config.map_save_path)
    reference_panorama = load_reference_map(config.map_save_path, config.matching_method)
    if reference_panorama is None:
        print("Reference map not found. Please run the script to create it first.")
    else:
        if(config.matching_method == MatchingMethod.SIFT):
            sensor_telemetry_pairs = calculate_R_align_from_reference_frames(reference_panorama,
                                                config.min_match_count,
                                                config.ransac_threshold,
                                                camMatrix,
                                                config.geometry_model)
        if(config.matching_method == MatchingMethod.SUPERGLUE):
            sensor_telemetry_pairs, frame_matching_statistics = calculate_R_align_from_reference_frames_lightglue(reference_panorama,
                                                config.min_match_count,
                                                config.ransac_threshold,
                                                camMatrix,
                                                config.geometry_model)

        
        R_align_final = find_R_align(sensor_telemetry_pairs)

        print("evaluate R align function, r_align_final: ")
        evaluate_R_align(R_align_final, sensor_telemetry_pairs)
        print("")

        print("Final Align Matrix: ", R_align_final)

        R_align_final, best_model_inliers, best_model_inliers_indices = run_ransac_for_alignment(sensor_telemetry_pairs, n = 4, k = 100, t_deg = ransac_angular_distance_threshold)

        print("evaluate R align function with Ransac, r_align_final: ")
        evaluate_R_align(R_align_final, sensor_telemetry_pairs)

        print("")
        print("evaluate R align function with Ransac on inlier pairs, r_align_final: ")
        evaluate_R_align(R_align_final, best_model_inliers)
        print("")
        print("Final Align Matrix: ", R_align_final)

        print("All pairs: ", len(sensor_telemetry_pairs))
        print("Inlier pairs: ", len(best_model_inliers))

        np.save(save_path, R_align_final)
        print("Align Matrix Saved at:" + save_path)

        print_matching_statistics_summary(frame_matching_statistics)


if __name__ == '__main__':
    # --- 1. SET UP ARGUMENT PARSER ---
    # This allows you to specify the config file from the command line
    parser = argparse.ArgumentParser(description="Calibrate R_align matrix from reference frames.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration INI file.')
    args = parser.parse_args()

    # --- 2. READ THE CONFIGURATION FILE ---
    config_parser = configparser.ConfigParser()
    config_parser.read(args.config)

    # --- 3. EXTRACT VALUES FROM CONFIG FILE ---
    # Paths section
    paths = config_parser['paths']
    save_path = paths.get('save_path')
    PATH_TO_REFERENCE_FOLDER = paths.get('reference_folder')
    PATH_TO_REFERENCE_MANIFEST = os.path.join(PATH_TO_REFERENCE_FOLDER, 'manifest.json')
    REFERENCE_PANORAMA_PATH = paths.get('reference_panorama_path')
    camera_intrinsics_path = paths.get('camera_intrinsics_path')
    save_matching_imgs_path = paths.get('save_matching_imgs_path')
    transformed_images_path = paths.get('transformed_images_path')
    reference_image_folder_temp = paths.get('reference_image_folder_temp')
    
    # Parameters section
    params = config_parser['parameters']
    number_minimum_matches = params.getint('min_match_count')
    ransac_threshold = params.getfloat('ransac_threshold')
    geometry_model = GeometryModel[params.get('geometry_model').upper()]
    feature_matching_method = MatchingMethod[params.get('feature_matching_method').upper()]
    num_features_deep_learning = params.getint('num_features_deep_learning')
    resize_scale_deep_learning = params.getfloat('resize_scale_deep_learning')
    ransac_angular_distance_threshold = params.getfloat('ransac_angular_distance_threshold')
    max_dimension = params.getint('max_dimension')
    visualize_matches = params.getboolean('visualize_matches')

    # Camera Intrinsics section
    cam_intr = config_parser['camera_intrinsics']
    to_estimate_camera_intrinsics = cam_intr.getboolean('to_estimate')
    width_intrinsics_path = cam_intr.getint('original_width')
    height_intrinsics_path = cam_intr.getint('original_height')

    # Intrinsics Estimation section
    est_intr = config_parser['intrinsics_estimation']
    estimate_camera_intrinsics_params = {
        'image_width': est_intr.getint('image_width'),
        'image_height': est_intr.getint('image_height'),
        'field_of_view_v': est_intr.getfloat('field_of_view_v'),
        'field_of_view_h': est_intr.getfloat('field_of_view_h')
    }

    # ROI (Region of Interest) section - parsing a list of floats
    roi_str = config_parser['roi'].get('crop_percentages')
    ROI = [float(x.strip()) for x in roi_str.split(',')]

    # --- 4. POPULATE THE AppConfig OBJECT ---
    config = AppConfig(
        reference_image_folder=PATH_TO_REFERENCE_FOLDER,
        reference_manifest_path=PATH_TO_REFERENCE_MANIFEST,
        reference_image_folder_temp=reference_image_folder_temp,
        query_image_folder_temp='',
        map_save_path=REFERENCE_PANORAMA_PATH,
        query_image_folder='',
        query_manifest_path='',
        min_match_count=number_minimum_matches,
        matching_method=feature_matching_method,
        ransac_threshold=ransac_threshold,
        visualize_matches=visualize_matches,
        save_matching_imgs_path=save_matching_imgs_path,
        transformed_images_path=transformed_images_path,
        camera_intrinsics_path=camera_intrinsics_path,
        geometry_model=geometry_model,
        to_estimate_camera_intrinsics=to_estimate_camera_intrinsics,
        estimate_camera_intrinsics_params=estimate_camera_intrinsics_params,
        roi=ROI,
        num_features_deep_learning=num_features_deep_learning,
        resize_scale_deep_learning=resize_scale_deep_learning,
        ransac_angular_distance_threshold=ransac_angular_distance_threshold,
        max_dimension=max_dimension,
        width_intrinsics_path=width_intrinsics_path,
        height_intrinsics_path=height_intrinsics_path
    )

    main(config, save_path)