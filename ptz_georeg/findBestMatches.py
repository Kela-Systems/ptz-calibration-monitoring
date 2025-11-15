import os

from .utils import load_reference_map, calculate_offsets_from_visual_matches, AppConfig

def main(PATH_TO_QUERY_IMAGES:str, PATH_TO_QUERY_MANIFEST:str, REFERENCE_PANORAMA_PATH:str):
    reference_panorama = load_reference_map(REFERENCE_PANORAMA_PATH)
    if reference_panorama is None:
        print("Reference map not found. Please run the script to create it first.")
    else:
        visual_telemetry_offsets = calculate_offsets_from_visual_matches(
                PATH_TO_QUERY_IMAGES,
                PATH_TO_QUERY_MANIFEST,
                reference_panorama
            )
        
        # --- Display the results ---
        print("\n--- Calculated Individual Offsets ---")
        if visual_telemetry_offsets:
            for offset in visual_telemetry_offsets:
                print("")
                # Assuming 'offset' is your dictionary of results
                print(
                f"""
                --------------------------------------------------
                Query Frame:         {offset['query_filename']}
                Best Match Ref:      {offset['best_match_ref_filename']}
                Number of Matches:   {offset['number_of_good_matches']}

                --- Telemetry Offsets ---
                Yaw:                 {offset['telemetry_offset_yaw']:.4f}
                Pitch:               {offset['telemetry_offset_pitch']:.4f}
                Roll:                {offset['telemetry_offset_roll']:.4f}

                --- Visual Offsets (from Homography) ---
                Yaw:                 {offset['visual_offset_yaw']:.4f}
                Pitch:               {offset['visual_offset_pitch']:.4f}
                Roll:                {offset['visual_offset_roll']:.4f}
                --------------------------------------------------
                """
                )
                print("")
        else:
            print("No valid offsets could be calculated.")


if __name__ == '__main__':
    PATH_TO_QUERY_IMAGES = 'C:/Users/vasil/Documents/data/PTZ_Calibration_and_georegistration/capture_output/capture_output/frames'
    PATH_TO_QUERY_MANIFEST = os.path.join(PATH_TO_QUERY_IMAGES, 'manifest.json')
    REFERENCE_PANORAMA_PATH = './data/referencePanorama.json'
    number_minimum_matches = 1000


    config = AppConfig(
        reference_image_folder="path/to/56_frames",
        reference_manifest_path="path/to/56_frames/manifest.json",
        query_image_folder="path/to/18_frames",
        query_manifest_path="path/to/18_frames/manifest.json",
    )

    main(PATH_TO_QUERY_IMAGES, PATH_TO_QUERY_MANIFEST, REFERENCE_PANORAMA_PATH)