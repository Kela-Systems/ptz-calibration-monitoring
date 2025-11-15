import os
from .utils import build_reference_map, save_reference_map, load_reference_map

def main(path_to_frames:str, path_to_manifest:str, output_path:str):
    reference_panorama = build_reference_map(path_to_frames, path_to_manifest)
    save_reference_map(reference_panorama, output_path)
    reference_panorama = load_reference_map(output_path)

if __name__ == '__main__':
    PATH_TO_IMAGES = 'C:/Users/vasil/Documents/data/PTZ_Calibration_and_georegistration/capture_output_56/capture_output_56/frames'
    PATH_TO_MANIFEST = os.path.join(PATH_TO_IMAGES, 'manifest.json')
    OUTPUT_PATH = './data/referencePanorama.json'
    main(PATH_TO_IMAGES, PATH_TO_MANIFEST, OUTPUT_PATH)