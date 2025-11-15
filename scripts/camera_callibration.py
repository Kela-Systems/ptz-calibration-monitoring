import os
import glob
import cv2 as cv
import numpy as np

import configparser
import argparse

def calculate_fov_from_intrinsics(K, image_width, image_height):
    """
    Calculates the horizontal and vertical Field of View (FoV) from a camera matrix.

    Args:
        K (np.ndarray): The 3x3 camera intrinsic matrix.
        image_width (int): The width of the images used for calibration.
        image_height (int): The height of the images used for calibration.

    Returns:
        tuple: A tuple containing (fov_h, fov_v) in degrees.
    """
    # Extract focal lengths in pixels from the camera matrix
    fx = K[0, 0]
    fy = K[1, 1]

    # Calculate FoV in radians
    fov_h_rad = 2 * np.arctan(image_width / (2 * fx))
    fov_v_rad = 2 * np.arctan(image_height / (2 * fy))

    # Convert radians to degrees
    fov_h_deg = np.rad2deg(fov_h_rad)
    fov_v_deg = np.rad2deg(fov_v_rad)

    return fov_h_deg, fov_v_deg

def calibrate(chessboardfolder: str, savePath: str, nRows: int, nCols: int, showPics: bool = True):
    """
    Performs camera calibration using checkerboard images.

    Args:
        chessboardfolder (str): Path to the directory containing checkerboard images.
        savePath (str): Path to save the calibration results (.npz file).
        nRows (int): Number of internal corners along one dimension of the checkerboard.
        nCols (int): Number of internal corners along the other dimension.
        showPics (bool): If True, displays the images with detected corners.
    """
    # --- 1. SETUP ---
    # Termination criteria for corner refinement
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...
    wordlPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    wordlPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

    # Arrays to store points from all the images.
    wordlPtsList = []  # 3D points in real world space
    imgPtsList = []    # 2D points in image plane.

    # Find all image files (BUG FIXED HERE, and added .png support)
    image_patterns = ['*.jpg', '*.png', '*.jpeg']
    imgPathList = []
    for pattern in image_patterns:
        imgPathList.extend(glob.glob(os.path.join(chessboardfolder, pattern)))
    
    if not imgPathList:
        print(f"Error: No images found in '{chessboardfolder}'. Check the path.")
        return None, None

    print(f"Found {len(imgPathList)} images. Processing...")
    
    # --- 2. FIND CORNERS ---
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        
        # Note: The order of nRows/nCols might need to be swapped depending on your board's orientation
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), None)

        if cornersFound:
            print(f"  Pattern found in: {os.path.basename(curImgPath)}")
            wordlPtsList.append(wordlPtsCur)
            
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
            
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nCols, nRows), cornersRefined, cornersFound)
                cv.imshow('Chessboard', imgBGR)
                key = cv.waitKey(0)
                if key == ord('q') or key == 27: # 27 is the keycode for ESC
                    print("  User aborted.")
                    break
    cv.destroyAllWindows()

    # --- 3. CALIBRATE ---
    # Add error handling for no valid images found
    if not wordlPtsList:
        print("\nError: Could not find the checkerboard pattern in any of the images.")
        print("Please check the grid size (nRows, nCols) and image quality.")
        return None, None

    print(f"\nUsing {len(wordlPtsList)} valid images for calibration.")
    
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(wordlPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    
    print('\n--- Calibration Results ---')
    print('Camera Matrix (K):\n', camMatrix)
    print('\nDistortion Coefficients (D):\n', distCoeff)
    print('Reproj Error (pixels): {:.4f}'.format(repError))
    
    # --- SAVE RESULTS ---
    output_dir = os.path.dirname(savePath)
    if output_dir: # Ensure it's not an empty string
        os.makedirs(output_dir, exist_ok=True)

    np.savez(savePath,
             camMatrix=camMatrix,
             distCoeff=distCoeff,
             repError=repError)
    print(f"\nCalibration data saved to: {savePath}")
    
    return camMatrix, distCoeff


if __name__ == '__main__':
    # --- 1. SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics using chessboard images.")
    parser.add_argument('--config', type=str, required=True, help='Path to the camera calibration INI file.')
    args = parser.parse_args()

    # --- 2. READ THE CONFIGURATION FILE ---
    config_parser = configparser.ConfigParser()
    config_parser.read(args.config)

    # --- 3. EXTRACT VALUES FROM CONFIG FILE ---
    # Paths section
    paths = config_parser['paths']
    PATH_TO_CHESSBOARD_FOLDER = paths.get('chessboard_folder')
    save_path = paths.get('save_path')

    # Chessboard section
    chessboard = config_parser['chessboard']
    nRows = chessboard.getint('rows')
    nCols = chessboard.getint('columns')

    # Visualization section
    visualization = config_parser['visualization']
    showPics = visualization.getboolean('show_pictures')

    # --- 4. RUN THE MAIN CALIBRATION FUNCTION ---
    print(f"Starting calibration with images from: {PATH_TO_CHESSBOARD_FOLDER}")
    print(f"Chessboard size: {nRows}x{nCols} inner corners.")
    print(f"Saving results to: {save_path}")
    
    calibrate(PATH_TO_CHESSBOARD_FOLDER, save_path, nRows, nCols, showPics)