import os
import numpy as np
import glob
from tqdm import tqdm
import cv2 as cv

def undistort_images(input_dir: str, output_dir: str, calibration_file: str):
    """
    Undistorts all images in a directory using pre-computed camera calibration data.

    Args:
        input_dir (str): Path to the directory containing the original (distorted) images.
        output_dir (str): Path to the directory where undistorted images will be saved.
        calibration_file (str): Path to the .npz file containing the camera matrix and
                                distortion coefficients.
    """
    # --- 1. Load Calibration Data ---
    try:
        calib_data = np.load(calibration_file)
        camMatrix = calib_data['camMatrix']
        distCoeff = calib_data['distCoeff']
    except FileNotFoundError:
        print(f"Error: Calibration file not found at '{calibration_file}'")
        return
    except KeyError:
        print(f"Error: Calibration file '{calibration_file}' is missing 'camMatrix' or 'distCoeff'.")
        return
        
    print("Successfully loaded camera calibration profile.")
    
    # --- 2. Prepare Input and Output Paths ---
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files in the input directory
    image_patterns = ['*.jpg', '*.png', '*.jpeg']
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not image_paths:
        print(f"Error: No images found in '{input_dir}'.")
        return

    print(f"Found {len(image_paths)} images to undistort. Starting process...")

    # --- 3. Loop Through Images and Undistort ---
    # Using tqdm for a user-friendly progress bar
    for img_path in tqdm(image_paths, desc="Undistorting Images"):
        # Read the raw (distorted) image
        raw_image = cv.imread(img_path)
        if raw_image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # Undistort the image
        # cv.undistort is a direct way to do this.
        # For potentially better results, you can use the optimal matrix method.
        # h, w = raw_image.shape[:2]
        # newCamMatrix, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (w,h), 1, (w,h))
        # undistorted_image = cv.undistort(raw_image, camMatrix, distCoeff, None, newCamMatrix)
        # x, y, w, h = roi
        # undistorted_image = undistorted_image[y:y+h, x:x+w] # Crop the image
        
        # For most cases, the direct method is sufficient and easier
        undistorted_image = cv.undistort(raw_image, camMatrix, distCoeff, None, None)

        # Create the full path for the output file
        base_filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, base_filename)

        # Save the undistorted image
        cv.imwrite(output_path, undistorted_image)

    print(f"\nUndistortion complete. {len(image_paths)} images saved to '{output_dir}'.")

if __name__ == '__main__':
    PATH_TO_INPUT_FOLDER = 'C:/Users/vasil/Documents/data/PTZ_Calibration_and_georegistration/chessboard_camera_recording_onvifcam-dev-1_20251008_173128/chessboard_camera_recording_onvifcam-dev-1_20251008_173128'
    PATH_TO_UNDISTORTED_IMAGES = './data/undistortedImages/'
    calibration_file = './camera_intrinsics/calibration.npz'
    undistort_images(PATH_TO_INPUT_FOLDER, PATH_TO_UNDISTORTED_IMAGES, calibration_file)