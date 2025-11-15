import json
import cv2
import os
import pickle
import math
from tqdm import tqdm
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
import cv2
import random
import shutil
from matplotlib import pyplot as plt
from kornia_moons.viz import *

from .match_info import MatchInfo, SensorTelemetryPair, FrameMatchingPointsStatistics

from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.optimize import minimize

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

# used
class MatchingMethod(Enum):
    """Enumeration for different feature matching methods."""
    SIFT = 0
    SUPERGLUE = 1
# used
class GeometryModel(Enum):
    """Enumeration for different feature matching methods."""
    HOMOGRAPHY = 0
    ESSENTIAL = 1

# used
@dataclass
class AppConfig:
    """
    Configuration class to store all arguments and settings for the application.
    """
    # --- Input Paths ---
    reference_image_folder: str
    reference_manifest_path: str
    query_image_folder: str
    query_manifest_path: str
    reference_image_folder_temp: str
    query_image_folder_temp: str

    # --- Output Paths ---
    map_save_path: str = "reference_map.pkl"

    # --- Feature Matching Parameters ---
    feature_method: str = "sift"  # Can be 'sift', 'orb', etc.
    min_match_count: int = 15

    # --- Telemetry Filtering Parameters ---
    use_telemetry_filter: bool = True
    yaw_threshold: float = 45.0
    pitch_threshold: float = 20.0

    # --- Homography and Decomposition Parameters ---
    ransac_threshold: float = 5.0
    max_acceptable_roll_error: float = 15.0 # For the 'roll-is-zero' prior
    max_acceptable_total_error: float = 45.0 # For the telemetry-guided prior

    # --- A place to store runtime data (optional but useful) ---
    reference_map: Optional[List[dict]] = field(default=None, repr=False)

    # --- Feature Matching Method ---
    # 0 for SIFT
    # 1 for superpoint-superglue
    matching_method: MatchingMethod = MatchingMethod.SIFT

    visualize_matches: bool = False

    save_matching_imgs_path: str = ''

    transformed_images_path: str = ''

    camera_intrinsics_path: str = ''

    r_align_path:str = ' '

    align_rotation_matrix:bool = True

    geometry_model: GeometryModel = GeometryModel.HOMOGRAPHY

    to_estimate_camera_intrinsics:bool = False
    estimate_camera_intrinsics_params: Dict[str, Any] = field(default_factory=dict)

    roi: List[float] = field(default_factory=list)

    num_features_deep_learning:int = 2048

    resize_scale_deep_learning:float = 0.5

    output_excel_path:str = ''

    ransac_angular_distance_threshold:float = 1.5

    max_dimension:int = 1024

    max_xy_norm: float  = 0.01

    min_overlapping_ratio:float = 0.8

    width_intrinsics_path:int = 1280

    height_intrinsics_path:int = 720

# used
def calculateKorniaFeatures_single(
    image_path: str,
    detector: K.feature.DISK,
    device: torch.device,
    camMatrix: np.ndarray,
    distCoeff: np.ndarray,
    num_features: int = 2048,
    mask: np.ndarray = None,
    resize_scale:float = 1.0
)->KF.DISKFeatures:
    """
    Loads a SINGLE image, UNDISTORTS it using OpenCV, converts to a tensor,
    and calculates features using a Kornia detector.
    
    Args:
        image_path (str): The path to the input image.
        detector (K.feature.DISK): An initialized Kornia DISK detector object.
        device (torch.device): The device to run inference on (e.g., 'cuda' or 'cpu').
        camMatrix (np.ndarray): The 3x3 camera intrinsic matrix from calibration.
        distCoeff (np.ndarray): The distortion coefficients from calibration.
        num_features (int): The maximum number of features to detect.
        mask (np.ndarray, optional): A grayscale (H, W) uint8 mask. 
                                     Features are kept where the mask is non-zero.
                                     Defaults to None (keep all features).
        resize_scale (float): Factor to resize the image by before feature detection. 
                              1.0 means no resize. 0.5 means half size.
        
    Returns:
        A tuple containing:
        - A Kornia feature object, or None if the image cannot be loaded.
        - The scale factor used for resizing (to map keypoints back).
    """
    print(f"Processing {image_path}...")
    try:
        raw_image_bgr = cv2.imread(image_path)
        if raw_image_bgr is None:
            raise FileNotFoundError

        undistorted_image_bgr = cv2.undistort(raw_image_bgr, camMatrix, distCoeff, None, None)

        if resize_scale != 1.0:
            h, w = undistorted_image_bgr.shape[:2]
            new_h, new_w = int(h * resize_scale), int(w * resize_scale)

            # Use INTER_AREA for quality downsampling
            resized_image_bgr = cv2.resize(undistorted_image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if mask is not None:
                # Use NEAREST neighbor for masks to avoid creating intermediate values
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                resized_image_bgr = undistorted_image_bgr

        # Convert from BGR (OpenCV) to RGB (Kornia/PyTorch).
        undistorted_image_rgb = cv2.cvtColor(resized_image_bgr, cv2.COLOR_BGR2RGB)

    except FileNotFoundError:
        print(f"Warning: Failed to load image at {image_path}. Skipping.")
        return None

    # Convert the NumPy array (H, W, C) to a PyTorch tensor.
    image_tensor = K.utils.image_to_tensor(undistorted_image_rgb, keepdim=False)
    
    # Normalize to [0, 1] and move to the correct device.
    # Kornia's detector expects a float tensor.
    image_tensor = image_tensor.to(torch.float32) / 255.0
    image_tensor = image_tensor.to(device)

    assert image_tensor.shape[0] == 1 and image_tensor.dim() == 4


    # ---  Run Inference with Kornia ---
    with torch.inference_mode():
        # The detector expects a batch, so we pass our (1, 3, H, W) tensor.
        # It returns a tuple of feature objects, one for each image in the batch.
        # We extract the single result with [0].
        num_features = int(num_features)
        features = detector(image_tensor, num_features, pad_if_not_divisible=True)[0]
        
    # --- Filter features based on the mask ---
    if mask is not None:
        # Get the keypoint coordinates (N, 2) tensor of (x, y)
        keypoints_xy = features.keypoints

        # Round coordinates to integers to use them as indices for the mask
        keypoints_int = torch.round(keypoints_xy).to(torch.long)

        # Ensure indices are within the mask's bounds to prevent errors
        h, w = mask.shape
        # Clamp values: x coords (dim 1) to [0, w-1], y coords (dim 0) to [0, h-1]
        keypoints_int[:, 0].clamp_(0, w - 1)
        keypoints_int[:, 1].clamp_(0, h - 1)

        # Create a boolean tensor indicating which keypoints to keep.
        # We need the mask on the same device as the keypoints.
        mask_tensor = torch.from_numpy(mask).to(device)
        
        # Look up the mask value at each keypoint's (y, x) location.
        # Keep if mask value is > 0.
        # NOTE: Mask is indexed (row, col) which is (y, x)
        keep_indices = mask_tensor[keypoints_int[:, 1], keypoints_int[:, 0]] > 0
        
        print(f"Original features: {len(features.keypoints)}. "
              f"Filtered features: {torch.sum(keep_indices).item()}")

        # Filter all components of the Features object using the boolean indices
        features = KF.DISKFeatures(
            keypoints=features.keypoints[keep_indices],
            descriptors=features.descriptors[keep_indices],
            detection_scores=features.detection_scores[keep_indices],
            # Add other attributes if your version of Kornia's Features has them
        )
    
    if resize_scale != 1.0:
        # Calculate the factor to scale coordinates back up
        scale_factor = 1.0 / resize_scale
        
        # Create a new Features object with the scaled keypoints.
        # Note that descriptors and scores are NOT scaled.
        features = KF.DISKFeatures(
            keypoints=features.keypoints * scale_factor,
            descriptors=features.descriptors,
            detection_scores=features.detection_scores
        )

    return features

# used
def normalize_angle_delta(delta):
    """
    Normalizes an angle difference to the range [-180, 180].

    Args:
        delta (float): The angle difference, which can be any value.

    Returns:
        float: The equivalent angle difference in the range [-180, 180].
    """
    
    normalized_delta = (delta + 180) % 360 - 180
    return normalized_delta


# used
def keypoints_to_list(keypoints):
    """Converts a tuple of cv2.KeyPoint objects to a list of dictionaries."""
    if keypoints is None:
        return []
    return [
        {
            "pt": kp.pt,
            "size": kp.size,
            "angle": kp.angle,
            "response": kp.response,
            "octave": kp.octave,
            "class_id": kp.class_id,
        }
        for kp in keypoints
    ]

# used
def kornia_features_to_list(features: KF.DISKFeatures) -> List[Dict[str, Any]]:
    """
    Converts a Kornia Features object (from DISK) into a list of dictionaries,
    mirroring the output format of the SIFT keypoint conversion function.

    Args:
        features: The feature object returned by a Kornia detector like DISK.

    Returns:
        A list of dictionaries, where each dictionary represents a keypoint.
    """
    if features is None:
        return []
    
    # 1. Move the relevant data from the GPU to the CPU and convert to NumPy.
    keypoints_np = features.keypoints.cpu().numpy()
    scores_np = features.detection_scores.cpu().numpy()
    
    # 2. Build the list of dictionaries using a list comprehension.
    #    We use 'None' for attributes that DISK does not compute.
    return [
        {
            "pt": tuple(pt),        # (x, y) coordinate tuple
            "size": None,           # DISK does not provide size
            "angle": None,          # DISK does not provide angle
            "response": float(score), # Kornia's 'score' is the response
            "octave": None,         # DISK does not provide octave
            "class_id": None,       # DISK does not provide class_id
        }
        for pt, score in zip(keypoints_np, scores_np)
    ]

# used
def list_to_keypoints(keypoint_list):
    """Converts a list of dictionaries back to a tuple of cv2.KeyPoint objects."""
    if not keypoint_list:
        return tuple()
        
    return tuple(
        cv2.KeyPoint(
            x=d["pt"][0],
            y=d["pt"][1],
            size=d["size"],       
            angle=d["angle"],     
            response=d["response"], 
            octave=d["octave"],   
            class_id=d["class_id"], 
        )
        for d in keypoint_list
    )

# used
def save_reference_map(reference_map, output_path):
    """Saves the reference map, correctly handling KeyPoint objects."""
    print(f"Saving reference map to {output_path}...")
    
    # Create a serializable version of the map
    serializable_map = []
    for frame_data in reference_map:
        serializable_frame = {
            'filename': frame_data['filename'],
            'telemetry': frame_data['telemetry'],
            # Convert KeyPoints to a list of dicts before saving
            'keypoints': frame_data['keypoints'],
            'descriptors': frame_data['descriptors'],
            'hw': frame_data['hw']
        }
        serializable_map.append(serializable_frame)

    with open(output_path, 'wb') as f:
        pickle.dump(serializable_map, f)
        
    print("Save complete.")


def save_features_to_txt(output_path, keypoints, descriptors):
    """
    Saves keypoints and descriptors to a text file in the COLMAP plain-text format.

    Args:
        output_path (str): Path to the output .txt file.
        keypoints (np.ndarray): Array of shape (N, 2) for (x, y) coordinates.
        descriptors (np.ndarray): Array of shape (N, D) for descriptors.
    """
    assert keypoints.shape[0] == descriptors.shape[0], \
        "Number of keypoints and descriptors must match."
    
    num_keypoints = keypoints.shape[0]
    if num_keypoints == 0:
        # If there are no features, create an empty file with a valid header
        descriptor_dim = 0 if descriptors.ndim < 2 else descriptors.shape[1]
        with open(output_path, 'w') as f:
            f.write(f"0 {descriptor_dim}\n")
        return

    descriptor_dim = descriptors.shape[1]
    
    # Ensure keypoints have 2 columns (x, y)
    if keypoints.shape[1] != 2:
        raise ValueError("Keypoints must be of shape (N, 2).")

    with open(output_path, 'w') as f:
        f.write(f"{num_keypoints} {descriptor_dim}\n")
        
        for i in range(num_keypoints):
            # Format is x, y, scale, orientation (we use 0 for scale/ori)
            kp_line = f"{int(keypoints[i, 0])} {int(keypoints[i, 1])} 0.0 0.0"
            desc_line = " ".join(map(lambda x: f"{x:.8f}", descriptors[i]))
            f.write(f"{kp_line} {desc_line}\n")


def save_map_to_colmap_txt(reference_map, output_path: str):
    """
    Processes a reference_map and saves the features for each frame
    to a separate .txt file in the COLMAP plain-text format.

    This function handles both SIFT-style (list of cv2.KeyPoint) and
    SUPERGLUE-style (PyTorch tensors) feature formats.

    Args:
        reference_map (list): A list of dictionaries, where each dict represents
                              a frame and its features.
        output_path (str): The directory where the .txt files will be saved.
    """
    if not reference_map:
        print("Warning: The reference map is empty or None. Nothing to save.")
        return

    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving COLMAP feature files to '{output_path}'...")

    # 2. Iterate through each frame in the map
    for frame_data in reference_map:
        filename = frame_data['filename']
        keypoints = frame_data['keypoints']
        descriptors = frame_data['descriptors']
        
        # --- Data Normalization Step ---
        # Convert different data types into standard NumPy arrays
        kps_np = None
        descs_np = None

        if isinstance(keypoints, list):
            # SIFT-style: list of cv2.KeyPoint objects
            if not keypoints: # Handle case with no keypoints
                 kps_np = np.empty((0, 2), dtype=np.float32)
                 descs_np = np.empty((0, descriptors.shape[1]), dtype=np.float32) if descriptors is not None else np.empty((0, 128))
            else:
                # Assuming keypoint objects have a 'pt' attribute (like cv2.KeyPoint)
                kps_np = np.array([kp['pt'] for kp in keypoints], dtype=np.float32)
                descs_np = descriptors # Already a numpy array
        
        elif isinstance(keypoints, torch.Tensor):
            # SUPERGLUE-style: PyTorch tensors
            # Tensors might be on GPU, so move to CPU and convert to NumPy
            # SuperGlue often has shape (1, N, 2) or (1, N, 256)
            kps_np = keypoints.squeeze(0).cpu().numpy()
            descs_np = descriptors.squeeze(0).cpu().numpy()

        else:
            print(f"Warning: Skipping '{filename}'. Unsupported keypoint format: {type(keypoints)}")
            continue
            
        # --- File Saving Step ---
        
        # 3. Create the specific output filename
        base_name = os.path.basename(filename)
        output_filename = base_name + '.txt'
        full_output_path = os.path.join(output_path, output_filename)
        
        # 4. Save the normalized features to the .txt file
        try:
            save_features_to_txt(full_output_path, kps_np, descs_np)
            print(f"  - Saved features for {filename} to {output_filename}")
        except Exception as e:
            print(f"Error saving features for {filename}: {e}")

    print("Finished saving all feature files.")


def save_reference_map_to_COLMAP_txts(input_path:str, output_path:str):
    if not os.path.exists(input_path):
        return None
    
    print(f"Loading reference map from {input_path}...")
    with open(input_path, 'rb') as f:
        serializable_map = pickle.load(f)
    
    save_map_to_colmap_txt(serializable_map, output_path)


#used
def load_reference_map(input_path:str, matching_method:MatchingMethod):
    """Loads a reference map and reconstructs KeyPoint objects."""
    if not os.path.exists(input_path):
        return None
    
    print(f"Loading reference map from {input_path}...")
    with open(input_path, 'rb') as f:
        serializable_map = pickle.load(f)
    print("Load complete. Reconstructing KeyPoints...")

    # Reconstruct the original map structure
    reference_map = []
    for frame_data in serializable_map:
        if(matching_method == MatchingMethod.SIFT):
            keypoints = list_to_keypoints(frame_data['keypoints'])
            reconstructed_frame = {
                'filename': frame_data['filename'],
                'telemetry': frame_data['telemetry'],
                # Convert list of dicts back to KeyPoint objects after loading
                'keypoints': keypoints,
                'descriptors': frame_data['descriptors']
            }
        if(matching_method == MatchingMethod.SUPERGLUE):
            keypoints_list = frame_data['keypoints']
            descriptors = frame_data['descriptors']
            # save reference data onto cpu
            device = torch.device('cpu')
            kps_tensor, descs_tensor = list_to_kornia_tensors(keypoints_list, descriptors, device)
            reconstructed_frame = {
                'filename': frame_data['filename'],
                'telemetry': frame_data['telemetry'],
                # Convert list of dicts back to KeyPoint objects after loading
                'keypoints': kps_tensor,
                'descriptors': descs_tensor,
                'hw': frame_data['hw']
            }

        reference_map.append(reconstructed_frame)
        
    print("Reconstruction complete.")
    return reference_map

# used
def calculateSIFTFeatures(image_path: str, sift: cv2.SIFT, camMatrix: np.ndarray, distCoeff: np.ndarray, mask: np.ndarray):
    """
    Loads an image, undistorts it using camera parameters, and computes SIFT features.

    Args:
        image_path (str): The path to the input image.
        sift (cv2.SIFT): An initialized SIFT detector object.
        camMatrix (np.ndarray): The 3x3 camera intrinsic matrix from calibration.
        distCoeff (np.ndarray): The distortion coefficients from calibration.

    Returns:
        tuple: A tuple containing (keypoints, descriptors). Returns (None, None) if
               the image cannot be loaded.
    """
    # 1. Load the raw (distorted) image in grayscale
    raw_image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if raw_image_gray is None:
        print(f"Warning: Failed to load image at {image_path}. Skipping.")
        return None, None

    # 2. THE CRITICAL NEW STEP: Undistort the image
    # This creates a new, geometrically correct image.
    undistorted_image = cv2.undistort(raw_image_gray, camMatrix, distCoeff, None, None)

    # 3. Detect features on the CLEAN, undistorted image
    keypoints, descriptors = sift.detectAndCompute(undistorted_image, mask=mask)
    
    return keypoints, descriptors

# used
def createDISKModel(device:torch.device):
    detector = KF.DISK.from_pretrained("depth").to(device).eval()
    return detector

#used
def build_reference_map(config:AppConfig, camMatrix:np.ndarray, distCoeff:np.ndarray) -> list:
    """
    Processes a folder of images to build a reference feature map.

    For each image, it extracts matching features and pairs them with telemetry
    data from a manifest.json file.

    Args:
        image_folder (str): Path to the folder containing the .png image files.
        manifest_path (str): Path to the manifest.json file.

    Returns:
        list: A list of dictionaries. Each dictionary represents a frame and contains:
              - 'filename' (str): The name of the image file.
              - 'telemetry' (dict): Contains 'yaw', 'pitch', and 'roll'.
              - 'keypoints' (tuple): A tuple of cv2.KeyPoint objects from SIFT.
              - 'descriptors' (np.ndarray): A NumPy array of SIFT descriptors.
    """
    image_folder: str = config.reference_image_folder_temp
    manifest_path: str = config.reference_manifest_path

    # 1. Load the entire manifest JSON file into a dictionary for easy lookup.
    try:
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return []
    
    # Initialize the detector. This is done once for efficiency.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(config.matching_method==MatchingMethod.SIFT):
        detector = cv2.SIFT.create()
    if(config.matching_method==MatchingMethod.SUPERGLUE):
        detector = createDISKModel(device)
    
    reference_map = []
    
    # Get a sorted list of image filenames to process in a consistent order.
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    print(f"Building reference map from {len(image_filenames)} frames...")
    
    # 3. Loop through each image file in the folder. tqdm provides a progress bar.
    for filename in tqdm(image_filenames, desc="Processing frames"):
        # Check if the current filename exists as a key in our manifest data.
        if filename not in manifest_data:
            print(f"Warning: No manifest entry found for {filename}. Skipping.")
            continue
            
        # 4. Construct the full path to the image and load it in grayscale.
        image_path = os.path.join(image_folder, filename)
        hw = get_image_hw(image_path)
        if hw is None:
            continue
        
        # mask construction
        mask = make_mask(hw, config.roi)

        keypoints, descriptors = None, None
        if(config.matching_method==MatchingMethod.SIFT):
            keypoints, descriptors = calculateSIFTFeatures(image_path, detector, camMatrix, distCoeff, mask)
            keypoints = keypoints_to_list(keypoints)
        if(config.matching_method==MatchingMethod.SUPERGLUE):
            features = calculateKorniaFeatures_single(image_path, detector, device, camMatrix, distCoeff, config.num_features_deep_learning, mask, config.resize_scale_deep_learning)
            if features is not None:
                keypoints = kornia_features_to_list(features)
                descriptors = features.descriptors.cpu().numpy()
            else:
                descriptors = None
                keypoints = None
        
        # If no keypoints are found, we can't use this image.
        if descriptors is None:
            print(f"Warning: No features found in {filename}. Skipping.")
            continue
            
        # 6. Extract the relevant telemetry data from the manifest.
        telemetry_entry = manifest_data[filename]['attitude']
        telemetry = {
            'yaw': telemetry_entry['yaw'],
            'pitch': telemetry_entry['pitch'],
            'roll': telemetry_entry['roll']
        }
        
        # 7. Store all the collected information for this frame.
        frame_data = {
            'filename': filename,
            'telemetry': telemetry,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'hw': hw
        }
        reference_map.append(frame_data)
        
    print(f"Successfully built reference map with {len(reference_map)} valid frames.")
    return reference_map

# used
def find_best_matches_light_glue(lg_matcher: KF.LightGlueMatcher, 
                               query_keypoints: torch.tensor, 
                               query_descriptors: torch.tensor, 
                               query_hw: List[int], 
                               reference_map: List[Dict[str, Any]],
                               camera_matrix: np.ndarray,
                               min_match_count=200,
                               ransac_threshold=5.0,
                               geometry_model:GeometryModel=GeometryModel.HOMOGRAPHY)->List[MatchInfo]:
    """
    Finds the best matching frame in the reference map for a given query using LightGlue.
    Assumes query data is on the GPU and reference data is on the CPU.

    Args:
        lg_matcher: The pre-initialized LightGlue model, already on the GPU.
        query_keypoints: A torch.Tensor of query keypoints, already on the GPU.
        query_descriptors: A torch.Tensor of query descriptors, already on the GPU.
        query_hw: The [height, width] of the query image.
        reference_map: A list of dictionaries, where features are torch.Tensors on the CPU.
        min_match_count: The minimum number of good matches required.

    Returns:
        A tuple containing:
            - dict: The best matching frame from the reference map.
            - np.ndarray: Matched keypoints from the query image (mkpts0) on the CPU.
            - np.ndarray: Matched keypoints from the best reference image (mkpts1) on the CPU.
        Returns (None, None, None) if no good match is found.
    """
    device = query_keypoints.device
    query_hw_tensor = torch.tensor(query_hw, device=device)

    best_matches:List[MatchInfo] = []

    # Create LAFs for the query keypoints with default scale and orientation
    # The [None] adds the batch dimension required by the laf function.
    query_lafs = KF.laf_from_center_scale_ori(
        query_keypoints[None], torch.ones(1, len(query_keypoints), 1, 1, device=device)
    )

    for ref_frame in reference_map:
        ref_kps_tensor_cpu = ref_frame['keypoints']
        ref_descs_tensor_cpu = ref_frame['descriptors']
        ref_hw = ref_frame['hw']

        ref_kps_tensor_gpu = ref_kps_tensor_cpu.to(device)
        ref_descs_tensor_gpu = ref_descs_tensor_cpu.to(device)
        ref_hw_tensor = torch.tensor(ref_hw, device=device)

        # Create LAFs for the reference keypoints
        ref_lafs = KF.laf_from_center_scale_ori(
            ref_kps_tensor_gpu[None], torch.ones(1, len(ref_kps_tensor_gpu), 1, 1, device=device)
        )

        with torch.inference_mode():
            dists, idxs = lg_matcher(query_descriptors, ref_descs_tensor_gpu, query_lafs, ref_lafs, hw1=query_hw_tensor, hw2=ref_hw_tensor)
            
            idxs = idxs.to(torch.device('cpu'))
            valid_rows_mask = (idxs[:, 0] != -1) & (idxs[:, 1] != -1)
            good_idxs = idxs[valid_rows_mask]
            num_current_matches = len(good_idxs)
            if(num_current_matches<min_match_count):
                continue

            query_keypoints = query_keypoints.to(torch.device('cpu'))
            matched_points_query = query_keypoints[good_idxs[:, 0]]
            matched_points_ref = ref_kps_tensor_cpu[good_idxs[:, 1]]

            cumulative_score = dists.sum().item()

            homography_matrix, nr_inliers, query_points_inliers, ref_points_inliers = calculate_homography_from_tensors(matched_points_query, 
                                                                                                                        matched_points_ref,
                                                                                                                        camera_matrix,
                                                                                                                        ransac_thresh=ransac_threshold,
                                                                                                                        geometry_model=geometry_model)

            if homography_matrix is not None:
                if(nr_inliers)>min_match_count:
                    matchInfo = MatchInfo()
                    matchInfo.cumulative_score = cumulative_score
                    matchInfo.ref_frame = ref_frame
                    matchInfo.homography_matrix = homography_matrix
                    matchInfo.num_raw_matches = num_current_matches
                    matchInfo.num_inliers = nr_inliers
                    matchInfo.mkpts_query = query_points_inliers
                    matchInfo.mkpts_ref = ref_points_inliers
                    best_matches.append(matchInfo)

    return best_matches

# used
def find_best_match(query_keypoints, query_descriptors, reference_map, camera_matrix:np.ndarray, min_match_count:int=15,ransac_threshold:float=5.0, geometry_model:GeometryModel = GeometryModel.HOMOGRAPHY)->List[MatchInfo]:
    """
    Finds the best matching frame in the reference map for a given query.

    Args:
        query_keypoints (tuple): Keypoints of the query image.
        query_descriptors (np.ndarray): SIFT descriptors for the query image.
        reference_map (list): The pre-computed reference feature map.
        min_match_count (int): The minimum number of good matches required to consider
                               a match valid.

    Returns:
        A tuple containing:
            - dict: The best matching frame from the reference map.
            - list: The list of good cv2.DMatch objects for the best pair.
            Returns (None, None) if no good match is found.
    """
    best_matches:List[MatchInfo] = []

    # Use a Brute-Force Matcher
    bf = cv2.BFMatcher()

    # Iterate through every frame in our reference map
    for ref_frame in reference_map:
        ref_descriptors = ref_frame['descriptors']
        matches = bf.knnMatch(query_descriptors, ref_descriptors, k=2)
        
        # Apply Lowe's ratio test to find good matches
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass

        if(len(good_matches)>min_match_count):
            ref_keypoints = ref_frame['keypoints']
            src_pts = np.float32([ query_keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
            dst_pts = np.float32([ ref_keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
            if(geometry_model==GeometryModel.HOMOGRAPHY):
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            else:
                # Calculate the Essential Matrix
                H, mask = cv2.findEssentialMat(
                src_pts,
                dst_pts,
                cameraMatrix=camera_matrix,  # This is the crucial new input
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0  # A good starting point, measured in pixels
                )

            inlier_mask = mask.ravel() == 1
            query_points_inliers = src_pts[inlier_mask]
            ref_points_inliers = dst_pts[inlier_mask]
            query_points_inliers = np.squeeze(query_points_inliers)
            ref_points_inliers = np.squeeze(ref_points_inliers)
            unique_inlier_ref_points = np.unique(ref_points_inliers, axis=0)
            unique_inlier_query_points = np.unique(query_points_inliers, axis=0)
            nr_unique_inlier_ref_points = len(unique_inlier_ref_points)
            nr_unique_inlier_query_points = len(unique_inlier_query_points)    
            # The 'mask' can be used to count inliers
            num_inliers = np.sum(mask)
            if(geometry_model==GeometryModel.HOMOGRAPHY):
                print(f"Homography calculated with {num_inliers}/{len(src_pts)} inliers.")
            if(geometry_model==GeometryModel.ESSENTIAL):
                print(f"Essential matrix calculated with {num_inliers}/{len(src_pts)} inliers.")
            if H is not None:
                if(num_inliers>min_match_count and nr_unique_inlier_ref_points>min_match_count and nr_unique_inlier_query_points>min_match_count):
                    matchInfo = MatchInfo()
                    matchInfo.cumulative_score = -1.0
                    matchInfo.ref_frame = ref_frame
                    matchInfo.homography_matrix = H
                    matchInfo.num_raw_matches = len(good_matches)
                    matchInfo.num_inliers = num_inliers
                    matchInfo.mkpts_query = query_points_inliers
                    matchInfo.mkpts_ref = ref_points_inliers
                    best_matches.append(matchInfo)
                else:
                    print(f"Matrix rejected with inliers: {num_inliers}, unique inlier ref points: {nr_unique_inlier_ref_points}, unique inlier query points: {nr_unique_inlier_query_points}")

    return best_matches

# used
def calculate_offsets_from_visual_matches(query_folder:str, 
                                          query_manifest_path:str, 
                                          reference_map, 
                                          min_match_count:int, 
                                          matching_method:MatchingMethod, 
                                          ransac_threshold:float, 
                                          camera_matrix:np.ndarray, 
                                          dist_coeff:np.ndarray,
                                          geometry_model:GeometryModel,
                                          r_align:np.ndarray,
                                          roi:List[float],
                                          num_of_deeplearning_featuers:int,
                                          resize_scale_deep_learning:float,
                                          max_xy_norm:float,
                                          min_overlapping_ratio:float):
    """
    Calculates the telemetry offsets for a set of query frames against a reference map.
    """
    # Load the query manifest file
    try:
        with open(query_manifest_path, 'r') as f:
            query_manifest = json.load(f)
    except FileNotFoundError:
        print(f"Error: Query manifest file not found at {query_manifest_path}")
        return []

    if(matching_method == MatchingMethod.SIFT):
        # Initialize SIFT detector and the list to store results
        sift_detector = cv2.SIFT.create()
    if(matching_method == MatchingMethod.SUPERGLUE):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        disk_detector = createDISKModel(device)
        lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)

    all_offsets = []

    pair_id = 0

    query_filenames = sorted([f for f in os.listdir(query_folder) if f.endswith('.png')])
    
    print(f"\nMatching {len(query_filenames)} query frames against the reference map...")
    for filename in tqdm(query_filenames, desc="Matching query frames"):
        if filename not in query_manifest:
            continue
            
        # Load and prepare the single query frame
        image_path = os.path.join(query_folder, filename)
        hw = get_image_hw(image_path)
        if hw is None:
            continue
        mask = make_mask(hw, roi) 
            
        if(matching_method == MatchingMethod.SIFT):
            query_kp, query_des = calculateSIFTFeatures(image_path=image_path,
                                  sift=sift_detector,
                                  camMatrix=camera_matrix,
                                  distCoeff=dist_coeff,
                                  mask=mask)
            if query_des is None:
                continue
            # Find the best visual match in the reference map
            best_matches = find_best_match(query_kp, query_des, reference_map, camera_matrix, min_match_count, ransac_threshold, geometry_model)
        if(matching_method==MatchingMethod.SUPERGLUE):
            query_features = calculateKorniaFeatures_single(image_path, 
                                                            disk_detector, 
                                                            device,
                                                            camera_matrix,
                                                            dist_coeff,
                                                            num_of_deeplearning_featuers,
                                                            mask,
                                                            resize_scale_deep_learning)
            if query_features is not None:
                disk_keypoints = query_features.keypoints
                disk_descriptors = query_features.descriptors
                hw = get_image_hw(image_path)
                best_matches = find_best_matches_light_glue(lg_matcher, 
                               disk_keypoints, 
                               disk_descriptors, 
                               hw, 
                               reference_map,
                               camera_matrix,
                                min_match_count, 
                                ransac_threshold,
                                geometry_model
                               )
            else:
                disk_descriptors = None
                disk_keypoints = None

        query_telemetry = query_manifest[filename]['attitude']
        
        for match in best_matches:
            if match.is_valid:
                pair_id = pair_id + 1
                test_sequences = ['yxz']
                for euler_sequence in test_sequences:
                    ref_telemetry = match.ref_frame['telemetry']
                    R_ref_telemetry = construct_R_from_telemetry(euler_sequence,
                                    ref_telemetry['yaw'],
                                    ref_telemetry['pitch'],
                                    ref_telemetry['roll']
                                )
                    
                    R_query_telemetry = construct_R_from_telemetry(euler_sequence,
                                    query_telemetry['yaw'],
                                    query_telemetry['pitch'],
                                    query_telemetry['roll']
                                )
                    
                    R_telemetry_query_ref = R_ref_telemetry @ R_query_telemetry.T

                    telemetry_offset_yaw = query_telemetry['yaw'] - ref_telemetry['yaw']
                    telemetry_offset_yaw = normalize_angle_delta(telemetry_offset_yaw)
                    telemetry_offset_pitch = query_telemetry['pitch'] - ref_telemetry['pitch']
                    telemetry_offset_pitch = normalize_angle_delta(telemetry_offset_pitch)
                    telemetry_offset_roll = query_telemetry['roll'] - ref_telemetry['roll']
                    telemetry_offset_roll = normalize_angle_delta(telemetry_offset_roll)

                    hw = get_image_hw(image_path)
                    h, w = hw[0], hw[1]
                    query_manifest_filename = query_manifest[filename]
                    offsets_yaw, offsets_pitch, offsets_roll, R_physicals, R_xy_norms = estimate_visual_offsets(match.homography_matrix,
                                                                                                        R_telemetry_query_ref, 
                                                                                                        camera_matrix,
                                                                                                        geometry_model,
                                                                                                        match.mkpts_query,
                                                                                                        match.mkpts_ref,
                                                                                                        r_align,
                                                                                                        euler_sequence)
                    offset_yaw = None
                    offset_pitch = None
                    offset_roll = None
                    R_physical = None
                    R_xy_norm = None
                    min_rxy_norm = 10000000000
                    for ind_pos, r_xy_norm in enumerate(R_xy_norms):
                        if(r_xy_norm < min_rxy_norm):
                            min_rxy_norm = r_xy_norm
                            offset_yaw = offsets_yaw[ind_pos]
                            offset_pitch = offsets_pitch[ind_pos]
                            offset_roll = offsets_roll[ind_pos]
                            R_physical = R_physicals[ind_pos]
                            R_xy_norm = R_xy_norms[ind_pos]                    

                    if (offset_yaw is not None) and (offset_pitch is not None) and (offset_roll is not None) and (R_physical is not None):
                        #R_offset = R_query_telemetry.T @ R_physical.T @ R_ref_telemetry
                        # we want to find the offset from reference to query
                        R_offset_transpose = R_ref_telemetry.T @ R_physical @ R_query_telemetry
                        yaw_offset_angle_difference, pitch_offset_angle_difference, roll_offset_angle_difference = rotation_matrix_to_euler(euler_sequence, R_offset_transpose)
                
                        if (yaw_offset_angle_difference is not None) and (pitch_offset_angle_difference is not None) and (roll_offset_angle_difference is not None) and (R_physical is not None):
                            angular_dist = calculate_angular_dist(R_physical, R_telemetry_query_ref)
                            #if(abs(yaw_offset_angle_difference)<10 and abs(pitch_offset_angle_difference)<10 and abs(roll_offset_angle_difference)<10):
                            
                            
                            image_query = cv2.imread(image_path)
                            image_query = cv2.undistort(image_query, camera_matrix, dist_coeff, None, None)
                            height, width, channels = image_query.shape
                            total_pixels = width * height
                            query_mask = np.full((height, width), 255, dtype=np.uint8)
                            warped_mask = cv2.warpPerspective(query_mask, match.homography_matrix, (width, height))
                            overlapping_pixel_count = cv2.countNonZero(warped_mask)
                            overlapping_ratio = overlapping_pixel_count/total_pixels
                            
                            if((R_xy_norm<max_xy_norm) and (overlapping_ratio>min_overlapping_ratio)):
                                offset = {
                                    'query_filename': filename,
                                    'best_match_ref_filename': match.ref_frame['filename'],
                                    'telemetry_offset_yaw': telemetry_offset_yaw,
                                    'telemetry_offset_pitch': telemetry_offset_pitch,
                                    'telemetry_offset_roll': telemetry_offset_roll,
                                    'physical_telemetry_offset_yaw': offset_yaw,
                                    'physical_telemetry_offset_pitch': offset_pitch,
                                    'physical_telemetry_offset_roll': offset_roll,
                                    'euler_angle_offset_yaw': yaw_offset_angle_difference,
                                    'euler_angle_offset_pitch': pitch_offset_angle_difference,
                                    'euler_angle_offset_roll': roll_offset_angle_difference,
                                    'number_of_good_matches': match.num_inliers,
                                    'matched_keypoints_ref': match.mkpts_ref,
                                    'matched_keypoints_query': match.mkpts_query,
                                    'homography_matrix': match.homography_matrix,
                                    'telemetry_rotation_matrix': R_telemetry_query_ref,
                                    'physical_rotation_matrix': R_physical,
                                    'angular distance': angular_dist,
                                    'euler sequence': euler_sequence,
                                    'xy norm': R_xy_norm,
                                    'pair id': pair_id,
                                    'overlapping ratio': overlapping_ratio
                                }
                                all_offsets.append(offset)
                            else:
                                print("")
                                print(f"  - Rejected Match of {match.ref_frame['filename']} onto {filename} with xy_norm {R_xy_norm:.2f} and overlapping ratio {overlapping_ratio:.2f}")
                                print("")
                        else:
                            print("")
                            print(f"  - Invalid match of {match.ref_frame['filename']} onto {filename}")
                            print("euler sequence: ", euler_sequence)
                            print("")
                    else:
                            print("")
                            print(f"  - Invalid match of {match.ref_frame['filename']} onto {filename}")
                            print("euler sequence: ", euler_sequence)
                            print("")
                    

    return all_offsets

# used
def calculate_homography_from_tensors(
    matching_query_points: torch.Tensor,
    matching_ref_points: torch.Tensor,
    camera_matrix: np.ndarray,
    min_match_count: int = 4,
    ransac_thresh: float = 5.0,
    geometry_model: GeometryModel = GeometryModel.HOMOGRAPHY
) -> Optional[np.ndarray]:
    """
    Calculates the homography matrix from matched PyTorch tensors.

    Args:
        matching_query_points: A tensor of shape (N, 2) with matched points
                               from the query image.
        matching_ref_points: A tensor of shape (N, 2) with corresponding matched
                             points from the reference image.
        min_match_count: The minimum number of points required to compute homography.
        ransac_thresh: The RANSAC reprojection threshold.

    Returns:
        np.ndarray: The 3x3 homography matrix H, or None if it cannot be computed.
    """

    # 1. We need at least 4 points to compute a homography
    if len(matching_query_points) < min_match_count:
        print(f"Warning: Not enough matches ({len(matching_query_points)}) to calculate homography. Need at least {min_match_count}.")
        return None, None, None, None

    # 2. Convert PyTorch Tensors to NumPy arrays.
    #    OpenCV functions expect NumPy arrays on the CPU.
    query_pts_np = matching_query_points.cpu().numpy()
    ref_pts_np = matching_ref_points.cpu().numpy()

    # 3. Calculate the homography matrix using RANSAC.
    #    RANSAC is a robust method that is not sensitive to a few remaining outliers.
    if(geometry_model==GeometryModel.HOMOGRAPHY):
        homography_matrix, mask = cv2.findHomography(
            query_pts_np, 
            ref_pts_np, 
            cv2.RANSAC, 
            ransac_thresh
        )
    if(geometry_model==GeometryModel.ESSENTIAL):
        query_pts_f32 = np.float32(query_pts_np)
        ref_pts_f32 = np.float32(ref_pts_np)
        # Calculate the Essential Matrix
        homography_matrix, mask = cv2.findEssentialMat(
            query_pts_f32,
            ref_pts_f32,
            cameraMatrix=camera_matrix,
            method=cv2.RHO,
            prob=0.999,
            threshold=1.0  # A good starting point, measured in pixels
        )

    
    # RANSAC can sometimes fail if all points are collinear.
    if homography_matrix is None:
        print("Warning: Homography calculation failed. The points might be collinear.")
        return None, None, None, None

    inlier_mask = mask.ravel() == 1
    
    query_points_inliers = query_pts_np[inlier_mask]
    ref_points_inliers = ref_pts_np[inlier_mask]    
    # The 'mask' can be used to count inliers
    num_inliers = np.sum(mask)

    print(f"Homography calculated with {num_inliers}/{len(query_pts_np)} inliers.")

    # You might want to add another check here for the number of inliers
    min_inlier_count = 4
    if num_inliers < min_inlier_count:
        print(f"Warning: Not enough inliers ({num_inliers}) to form a reliable homography.")
        return None, None, None, None
        
    return homography_matrix, num_inliers, query_points_inliers, ref_points_inliers

# used
def estimate_camera_intrinsics(image_width, image_height, fov_v_degrees, fov_h_degrees):
    """
    Estimates the camera intrinsic matrix K from both vertical and horizontal FoV.

    Args:
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        fov_v_degrees (float): Vertical field of view in degrees.
        fov_h_degrees (float): Horizontal field of view in degrees.
        
    Returns:
        np.ndarray: The 3x3 camera intrinsic matrix K.
    """
    # Principal point is assumed to be the center of the image
    cx = image_width / 2
    cy = image_height / 2

    # --- Calculate fy from fov_v ---
    fov_v_rad = math.radians(fov_v_degrees)
    # Add a small epsilon to avoid division by zero if fov is 180
    fy = cy / (math.tan(fov_v_rad / 2) + 1e-9)
    
    # --- Calculate fx from fov_h ---
    fov_h_rad = math.radians(fov_h_degrees)
    fx = cx / (math.tan(fov_h_rad / 2) + 1e-9)

    # (Optional but recommended) Sanity check: log a warning if fx and fy are very different
    if not np.isclose(fx, fy, rtol=0.05): # Check if they differ by more than 5%
        print(f"Warning: Calculated fx ({fx:.2f}) and fy ({fy:.2f}) differ significantly. "
              "This may indicate inconsistent FoV data.")

    # Construct the intrinsic matrix using both calculated values
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    
    return K

# used
def calculate_angular_dist(R1, R2):
    R_error = R1 @ R2.T
    trace = np.trace(R_error)
    angle_rad = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))
    angular_dist = np.degrees(angle_rad)
    return angular_dist

# used
def extract_best_rotation_with_telemetry(H, K, R_telemetry_ij):
    """
    Decomposes H and selects the best rotation by finding the solution closest
    to the offsets predicted by telemetry.

    Args:
        H (np.ndarray): The 3x3 homography matrix.
        K (np.ndarray): The 3x3 camera intrinsic matrix.
        telemetry_offsets (dict): A dict with keys 'yaw', 'pitch', 'roll' for the
                                  telemetry-predicted offsets.

    Returns:
        np.ndarray: The single best 3x3 rotation matrix, or None.
    """
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    if not rotations:
        return None

    best_R = None
    min_angular_dist = float('inf')

    # Iterate through each of the 4 possible solutions
    for R in rotations:
        angular_dist = calculate_angular_dist(R, R_telemetry_ij)
        if angular_dist < min_angular_dist:
            min_angular_dist = angular_dist
            best_R = R
    
    return best_R, min_angular_dist

def extract_rotation_positive_z_normal(H, K, R_telemetry_ij):
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    if not rotations:
        return None

    best_Rs = []
    best_Rs_angular_distances = []
    best_Rs_z_normals = []
    best_Rs_xy_norm = []

    # Iterate through each of the 4 possible solutions
    for (R, n, t) in zip(rotations, normals, translations):
        z_normal = n[2]
        if(z_normal>=0.0):
            tx, ty = t[0][0], t[1][0]
            xy_norm = np.sqrt(tx**2 + ty**2)
            best_Rs_xy_norm.append(xy_norm)
            best_Rs.append(R)
            ang_distance_telemetry = calculate_angular_dist(R, R_telemetry_ij)
            best_Rs_angular_distances.append(ang_distance_telemetry)
            best_Rs_z_normals.append(z_normal) 
    
    return best_Rs, best_Rs_angular_distances, best_Rs_z_normals, best_Rs_xy_norm

# used
def rotation_matrix_to_euler(telemetry_sequence,R):
    """
    Converts a rotation matrix to Euler angles (yaw, pitch, roll).
    The convention used here is ZYX, which corresponds to yaw, pitch, roll.
    """
    try:
        r = ScipyRotation.from_matrix(R)
        # Using 'zyx' order for yaw (Z), pitch (Y), roll (X). Angles are in degrees.
        angles = r.as_euler(telemetry_sequence, degrees=True)
        yaw = - angles[0]
        pitch = - angles[1]
        roll = angles[2]
        return yaw, pitch, roll
    except Exception as e:
        print(f"Error converting rotation matrix: {e}")
        return None, None, None

# used
def estimate_visual_offsets(H, 
                            R_telemetry_ij, 
                            camera_matrix,
                            geometry_model:GeometryModel.HOMOGRAPHY,
                            query_pts_f32,
                            ref_pts_f32,
                            R_align,
                            euler_sequence):
    visual_offsets_yaw, visual_offsets_pitch, visual_offsets_roll = [], [], []
    R_physicals = []
    R_xy_norms = []

    if H is not None:
        if(geometry_model==GeometryModel.HOMOGRAPHY):
            best_solution = extract_rotation_positive_z_normal(H, camera_matrix, R_telemetry_ij)
            if(best_solution is None):
                return [], [], [], []
            R_sensors = best_solution[0]
            xy_norms = best_solution[-1]
            for (R_sensor, xy_norm) in zip(R_sensors, xy_norms):
                if(R_sensor is None):
                    return [], [], [], []
                R_physical = R_align.T @ R_sensor @ R_align
                R_physicals.append(R_physical)
                R_xy_norms.append(xy_norm)
        if(geometry_model==GeometryModel.ESSENTIAL):
            # Decompose the Essential Matrix to get the R and t
            # This function internally handles the 4-way ambiguity and returns the single best solution.
            _, R_sensor, t, mask = cv2.recoverPose(
                H,
                query_pts_f32,
                ref_pts_f32,
                cameraMatrix=camera_matrix
            )
            R_physical = R_align.T @ R_sensor @ R_align

        for R_physical in R_physicals:
            if R_physical is not None:
                R_offset = R_physical @ R_telemetry_ij.T
                visual_offset_yaw, visual_offset_pitch, visual_offset_roll = rotation_matrix_to_euler(euler_sequence, R_offset)
                visual_offsets_yaw.append(visual_offset_yaw)
                visual_offsets_pitch.append(visual_offset_pitch)
                visual_offsets_roll.append(visual_offset_roll)
        return visual_offsets_yaw, visual_offsets_pitch, visual_offsets_roll, R_physicals, R_xy_norms
    else:
        return [], [], [], []

#used
def calculate_final_offset(all_offsets):
    """
    Calculates the final system offset using robust statistical methods.
    """
    
    if not all_offsets:
        print("Warning: No high-confidence matches found. Cannot calculate final offset.")
        return None

    print(f"\nCalculating final offset from {len(all_offsets)} high-confidence matches.")

    # 2. Calculate the system offset for each clean result
    angular_distances = []
    system_offsets_yaw = []
    system_offsets_pitch = []
    system_offsets_roll = []
    weights = []
    avg_weights = []
    sum_weights = 0.0
    eps = 0.00001

    for offset in all_offsets:
        angular_distances.append(offset['angular distance'])
        system_offsets_yaw.append(offset['euler_angle_offset_yaw'])
        system_offsets_pitch.append(offset['euler_angle_offset_pitch'])
        system_offsets_roll.append(offset['euler_angle_offset_roll'])
        w = offset['number_of_good_matches']
        weights.append(w)
        sum_weights = sum_weights + w

    for weight in weights:
        if(sum_weights>0):
            avg_weight = weight/sum_weights
            avg_weights.append(avg_weight)

    # Calculate and report the final estimates
    # Method A: Median
    median_yaw = np.median(system_offsets_yaw)
    median_pitch = np.median(system_offsets_pitch)
    median_roll = np.median(system_offsets_roll)

    # Method B: Weighted Mean (Potentially More Accurate)
    weighted_mean_yaw = np.average(system_offsets_yaw, weights=avg_weights)
    weighted_mean_pitch = np.average(system_offsets_pitch, weights=avg_weights)
    weighted_mean_roll = np.average(system_offsets_roll, weights=avg_weights)

    # system offset std
    std_yaw_offset = np.std(system_offsets_yaw)
    std_pitch_offset = np.std(system_offsets_pitch)
    std_roll_offset = np.std(system_offsets_roll)

    mean_angular_distances = np.mean(angular_distances)

    final_results = {
        "median_offset": {"yaw": median_yaw, "pitch": median_pitch, "roll": median_roll},
        "weighted_mean_offset": {"yaw": weighted_mean_yaw, "pitch": weighted_mean_pitch, "roll": weighted_mean_roll},
        "number_of_inlier_frames": len(all_offsets),
        "std":{"yaw": std_yaw_offset, "pitch": std_pitch_offset, "roll": std_roll_offset},
        "mean angular distance": mean_angular_distances
    }

    return final_results

# used
def euler_to_rotation_matrix(yaw, pitch, roll):
    """Converts Euler angles (yaw, pitch, roll) to a 3x3 rotation matrix."""
    r = ScipyRotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    return r.as_matrix()

# used
def test_homography_roundtrip(query_image, ref_image, K_query, K_ref, telemetry_offset):
    """
    Performs a round-trip test on the homography decomposition and reconstruction.
    """
    print("--- Starting Homography Round-Trip Test ---")
    h, w = query_image.shape[:2]

    # --- Step 1: Measure the "Ground Truth" Homography (H_measured) ---
    sift = cv2.SIFT.create()
    kp1, des1 = sift.detectAndCompute(query_image, None)
    kp2, des2 = sift.detectAndCompute(ref_image, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 15:
        print("Test failed: Not enough good matches to start.")
        return

    query_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H_measured, _ = cv2.findHomography(query_pts, ref_pts, cv2.RANSAC, 5.0)
    print("Step 1: Successfully calculated H_measured from feature points.")

    rotation_matrix = extract_best_rotation_with_telemetry(H_measured, K_ref, telemetry_offset)
    
    # Let's see the estimated motion
    yaw, pitch, roll = rotation_matrix_to_euler(rotation_matrix)
    print(f"Step 2: Decomposed motion (Y,P,R): ({yaw:.2f}, {pitch:.2f}, {roll:.2f})")
    pass

    R_warp = euler_to_rotation_matrix(yaw, pitch, roll)
    K_query_inv = np.linalg.inv(K_query)
    H_reconstructed = K_ref @ R_warp @ K_query_inv

    H_measured_norm = H_measured / H_measured[2, 2]
    H_reconstructed_norm = H_reconstructed / H_reconstructed[2, 2]
    
    difference = np.linalg.norm(H_measured_norm - H_reconstructed_norm)
    print(f"Difference between H_measured and H_reconstructed: {difference:.6f}")
    if difference < 1e-3:
        print("SUCCESS: The matrices are nearly identical.")
    else:
        print("WARNING: Matrices differ significantly. Check the math/conventions.")
    
# used
def list_to_kornia_tensors(
    keypoint_list: List[Dict[str, Any]], 
    descriptors: np.ndarray, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a standardized list of keypoint dictionaries and a NumPy array of 
    descriptors back into PyTorch Tensors on the specified device.

    This is the reverse operation needed before matching with LightGlue.
    """
    if not keypoint_list:
        # Return empty tensors with the correct shape if the list is empty
        keypoints_tensor = torch.empty((0, 2), dtype=torch.float32, device=device)
        descriptors_tensor = torch.empty((0, 128), dtype=torch.float32, device=device)
        return keypoints_tensor, descriptors_tensor

    # 1. Extract the (x, y) coordinates from the list of dictionaries
    points = [d['pt'] for d in keypoint_list]
    
    # 2. Convert the list of points to a NumPy array, then to a PyTorch tensor
    keypoints_tensor = torch.from_numpy(np.array(points, dtype=np.float32)).to(device)
    
    # 3. Convert the NumPy descriptors array to a PyTorch tensor
    #    (It's assumed descriptors is already a float32 NumPy array)
    descriptors_tensor = torch.from_numpy(descriptors).to(device)
    
    return keypoints_tensor, descriptors_tensor

# used
def get_image_hw(image_path: str) -> Optional[List[int]]:
    """
    Reads an image from the given path and returns its dimensions.

    Args:
        image_path: The full path to the image file.

    Returns:
        A list containing [height, width] of the image,
        or None if the image cannot be read.
    """
    # cv2.imread() is very fast for reading image headers and shape.
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Warning: Could not read image at path: {image_path}")
        return None
    
    # The .shape attribute of a NumPy array loaded by OpenCV gives
    # (height, width, channels) for a color image, or
    # (height, width) for a grayscale image.
    # Slicing with [:2] robustly gets the height and width in both cases.
    height, width = image.shape[:2]
    
    return [height, width]

# used
def visualize_matches_custom(
    query_image_path: str,
    ref_image_path: str,
    query_inlier_points_np: np.ndarray,
    ref_inlier_points_np: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeff: np.ndarray,
    save_path: str = "custom_visualization.png",
    linewidth: float = 0.0,
    draw_points: bool = True
):
    """
    Visualizes inlier matches with unique colors using a robust, custom
    Matplotlib and OpenCV implementation.
    """
    
    img1_bgr = cv2.imread(query_image_path)
    img2_bgr = cv2.imread(ref_image_path)
    if img1_bgr is None or img2_bgr is None:
        print(f"Error: Could not load images for visualization.")
        return
        
    undistorted_img1_bgr = cv2.undistort(img1_bgr, camera_matrix, dist_coeff, None, None)
    undistorted_img2_bgr = cv2.undistort(img2_bgr, camera_matrix, dist_coeff, None, None)
    
    img1_rgb = cv2.cvtColor(undistorted_img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(undistorted_img2_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
    
    ax1, ax2 = axes
    
    ax1.imshow(img1_rgb)
    ax1.set_title(os.path.basename(query_image_path))
    ax1.axis('off') # Hide axis ticks

    ax2.imshow(img2_rgb)
    ax2.set_title(os.path.basename(ref_image_path))
    ax2.axis('off')

    fig.subplots_adjust(wspace=0, hspace=0)

    num_matches = len(query_inlier_points_np)
    if num_matches == 0:
        print("No inliers to visualize.")
        plt.close(fig)
        return

    cmap = plt.get_cmap("nipy_spectral", num_matches)
    colors = [cmap(i) for i in range(num_matches)]
    np.random.shuffle(colors)

    if linewidth > 0:
        for i in range(num_matches):
            pt1 = query_inlier_points_np[i]
            pt2 = ref_inlier_points_np[i]

            con = plt.ConnectionPatch(
                xyA=pt2, xyB=pt1,
                coordsA="data", coordsB="data",
                axesA=ax2, axesB=ax1,
                color=colors[i],
                linewidth=linewidth,
                alpha=0.7
            )
            ax2.add_artist(con)
    if draw_points:
        for i in range(num_matches):
            pt1 = query_inlier_points_np[i]
            pt2 = ref_inlier_points_np[i]
            ax1.plot(pt1[0], pt1[1], 'o', color=colors[i], markersize=3)
            ax2.plot(pt2[0], pt2[1], 'o', color=colors[i], markersize=3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved custom visualization to {save_path}")

# used
def visualize_transformations(homography_matrix, 
                              image_A_path:str,
                              image_B_path:str,
                              save_path_side:str,
                              save_path_overlay:str,
                              camMatrix:np.ndarray,
                              distCoeff:np.ndarray):
    if homography_matrix is not None:
        image_A = cv2.imread(image_A_path)
        image_B = cv2.imread(image_B_path)
        image_A = cv2.undistort(image_A, camMatrix, distCoeff, None, None)
        image_B = cv2.undistort(image_B, camMatrix, distCoeff, None, None)
        height, width, channels = image_B.shape
        warped_image_A = cv2.warpPerspective(image_A, homography_matrix, (width, height))
        display_A = cv2.resize(warped_image_A, (width // 2, height // 2))
        display_B = cv2.resize(image_B, (width // 2, height // 2))
        side_by_side = np.hstack([display_A, display_B])
        cv2.imwrite(save_path_side, side_by_side)

        blended_image = cv2.addWeighted(warped_image_A, 0.5, image_B, 0.5, 0)
        print(f"Saved side-by-side comparison to: {save_path_side}")
        print(f"Saved blended overlay to: {save_path_overlay}")
        cv2.imwrite(save_path_overlay, blended_image)

# used
def visualize_epilines(
    query_img_path: str,
    ref_img_path: str,
    inlier_query_pts: np.ndarray, 
    inlier_ref_pts: np.ndarray,  
    E: np.ndarray,
    K: np.ndarray,
    save_path: str
):
    """
    Draws epipolar lines on a pair of images to visualize the correctness
    of the PROVIDED Essential Matrix E and Camera Matrix K, using pre-filtered
    inlier points.
    """
    # Load images in color
    img1 = cv2.imread(query_img_path)
    img2 = cv2.imread(ref_img_path)
    if img1 is None or img2 is None:
        print("Error loading images for epiline visualization.")
        return

    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    pts1 = np.int32(np.squeeze(inlier_query_pts))
    pts2 = np.int32(np.squeeze(inlier_ref_pts))

    lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines1 = lines1.reshape(-1, 3)
    img_ref_with_lines, _ = draw_lines(img2, img1, lines1, pts2, pts1)

    lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines2 = lines2.reshape(-1, 3)
    img_query_with_lines, _ = draw_lines(img1, img2, lines2, pts1, pts2)

    final_image = np.hstack((img_query_with_lines, img_ref_with_lines))
    
    cv2.imwrite(save_path, final_image)
    print(f"Saved epipolar line visualization to: {save_path}")

# used
def draw_lines(img1, img2, lines, pts1, pts2):
    r, c, _ = img1.shape
    img1_color = img1.copy()
    
    # Draw a random subset of lines for clarity (e.g., up to 20)
    num_points = len(pts1)
    indices = np.arange(num_points)
    np.random.shuffle(indices)
    
    for i in indices[:min(20, num_points)]: # Draw up to 20 random lines
        color = tuple(np.random.randint(0, 255, 3).tolist())
        r_line, pt1_point = lines[i], pts1[i]
        
        x0, y0 = map(int, [0, -r_line[2] / r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0] * c) / r_line[1]])
        
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 2)
        # Draw the corresponding point on the same image
        img1_color = cv2.circle(img1_color, tuple(pt1_point), 5, color, -1)
    
    return img1_color, None

# used
def construct_R_from_telemetry(euler_sequence, pan_deg, pitch_deg, roll_deg):
    """
    Constructs an absolute rotation matrix from telemetry data.
    
    Args:
        pan_deg (float): Yaw angle from telemetry. Positive is "turning right".
        pitch_deg (float): Pitch angle from telemetry. Positive is "looking up".
        roll_deg (float): Roll angle from telemetry.
        
    Returns:
        np.ndarray: A 3x3 rotation matrix representing the camera's orientation.
    """
    yaw_corrected = -pan_deg 
    pitch_corrected = -pitch_deg
    roll_corrected = roll_deg
    
    rotation = ScipyRotation.from_euler(
        euler_sequence, 
        [yaw_corrected, pitch_corrected, roll_corrected], 
        degrees=True
    )
    
    return rotation.as_matrix()

# used
def calculate_R_align_from_reference_frames(reference_map:List[Dict[str, Any]],
                                            min_match_count:int,
                                            ransac_threshold:float,
                                            camera_matrix:np.ndarray,
                                            geometry_model:GeometryModel
                                            )->List[SensorTelemetryPair]:
    """
    Calculates the R align matrix with matching pairs of reference frames.
    """
    sensorTelemetryPairs:List[SensorTelemetryPair] = []
    euler_sequence = 'yxz'
    
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    
    number_of_references = len(reference_map)
    for ref_i in range(number_of_references):
        i_frame = reference_map[ref_i]
        ref_i_descriptors = i_frame['descriptors']
        ref_i_keypoints = i_frame['keypoints']
        telemetry_i = i_frame['telemetry']
        print('')
        print("Finding matches for frame: ", i_frame['filename'])
        for ref_j in range(ref_i+1,number_of_references):
            j_frame = reference_map[ref_j]
            ref_j_descriptors = j_frame['descriptors']
            if i_frame['descriptors'] is None or j_frame['descriptors'] is None:
                continue
            ref_j_keypoints = j_frame['keypoints']

            matches = bf_matcher.knnMatch(ref_i_descriptors, ref_j_descriptors, k=2)

            # Apply Lowe's ratio test to find good matches
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            except ValueError:
                pass

            if(len(good_matches)>min_match_count):
                print("Found match: ", j_frame['filename'])

                i_pts = np.float32([ ref_i_keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                j_pts = np.float32([ ref_j_keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                if(geometry_model==GeometryModel.HOMOGRAPHY):
                    H_ij, mask = cv2.findHomography(i_pts, j_pts, cv2.RANSAC, ransac_threshold)
                    if H_ij is None:
                        print(f"Warning: Could not find Homography for frames {ref_i}-{ref_j}. Skipping pair.")
                        continue
                if(geometry_model==GeometryModel.ESSENTIAL):
                    E_ij, mask = cv2.findEssentialMat(
                                i_pts, 
                                j_pts, 
                                cameraMatrix=camera_matrix, 
                                method=cv2.RANSAC, 
                                prob=0.999, 
                                threshold=ransac_threshold # threshold in pixels
                            )
                    if E_ij is None:
                        print(f"Warning: Could not find Essential Matrix for frames {ref_i}-{ref_j}. Skipping pair.")
                        continue


                inlier_mask = mask.ravel() == 1
                i_points_inliers = i_pts[inlier_mask]
                j_points_inliers = j_pts[inlier_mask]
                i_points_inliers = np.squeeze(i_points_inliers)
                j_points_inliers = np.squeeze(j_points_inliers)
                unique_inlier_i_points = np.unique(i_points_inliers, axis=0)
                unique_inlier_j_points = np.unique(j_points_inliers, axis=0)
                nr_unique_inlier_i_points = len(unique_inlier_i_points)
                nr_unique_inlier_j_points = len(unique_inlier_j_points)    
                # The 'mask' can be used to count inliers
                num_inliers = np.sum(mask)
                if(num_inliers>min_match_count and nr_unique_inlier_i_points>min_match_count and nr_unique_inlier_j_points>min_match_count):
                    telemetry_j = j_frame['telemetry']
                    R_i_world = construct_R_from_telemetry(euler_sequence,
                                telemetry_i['yaw'],
                                telemetry_i['pitch'],
                                telemetry_i['roll']
                            )
                    R_j_world = construct_R_from_telemetry(euler_sequence,
                                telemetry_j['yaw'],
                                telemetry_j['pitch'],
                                telemetry_j['roll']
                            )
                    R_telemetry_ij = R_j_world @ R_i_world.T

                    if(geometry_model==GeometryModel.HOMOGRAPHY):
                        R_sensor_ij, angular_dist = extract_best_rotation_with_telemetry(H_ij, camera_matrix, R_telemetry_ij)
                    else:
                        retval, R_sensor_ij, t_sensor_ij, mask_RP = cv2.recoverPose(
                                    E_ij, i_pts, j_pts, cameraMatrix=camera_matrix, mask=mask
                                )
                        
                        angular_dist = calculate_angular_dist(R_sensor_ij, R_telemetry_ij)

                    if(angular_dist<10):
                        sensorTelemetryPair = SensorTelemetryPair()
                        sensorTelemetryPair.sensor_rotation_matrix = R_sensor_ij
                        sensorTelemetryPair.telemetry_rotation_matrix = R_telemetry_ij
                        if(geometry_model==GeometryModel.ESSENTIAL):
                            sensorTelemetryPair.sensor_translation_matrix = t_sensor_ij
                            sensorTelemetryPair.telemetry_translation_matrix = np.array([[0.0], [0.0], [0.0]])
                        sensorTelemetryPairs.append(sensorTelemetryPair)

                        print("--- Robust Rotation Matrices ---")
                        if(geometry_model==GeometryModel.HOMOGRAPHY):
                            print("R_sensor_ij (from Homography Decomposition):\n", np.round(R_sensor_ij, 4))
                        if(geometry_model==GeometryModel.ESSENTIAL):
                            print("R_sensor_ij (from Essential Decomposition):\n", np.round(R_sensor_ij, 4))
                        print("\nR_telemetry_ij (from Ideal Data):\n", np.round(R_telemetry_ij, 4))
                        print("\nAngular distance: \n", np.round(angular_dist, 4))
                        print("\nNumber of matches: \n", num_inliers)
                    else:
                        print("--- Rejected Rotation Matrices ---")
                        if(geometry_model==GeometryModel.HOMOGRAPHY):
                            print("R_sensor_ij (from Homography Decomposition):\n", np.round(R_sensor_ij, 4))
                        if(geometry_model==GeometryModel.ESSENTIAL):
                            print("R_sensor_ij (from Essential Decomposition):\n", np.round(R_sensor_ij, 4))
                        print("\nR_telemetry_ij (from Ideal Data):\n", np.round(R_telemetry_ij, 4))
                        print("\nAngular distance: \n", np.round(angular_dist, 4))
                        print("\nNumber of matches: \n", num_inliers)                        
    print("Number of matches found: ", len(sensorTelemetryPairs))        
    return sensorTelemetryPairs

# used
def calculate_R_align_from_reference_frames_lightglue(reference_map:List[Dict[str, Any]],
                                            min_match_count:int,
                                            ransac_threshold:float,
                                            camera_matrix:np.ndarray,
                                            geometry_model:GeometryModel
                                            )-> Tuple[List[SensorTelemetryPair], List[FrameMatchingPointsStatistics]]:
    """
    Calculates the R align matrix with matching pairs of reference frames.
    """
    sensorTelemetryPairs:List[SensorTelemetryPair] = []
    euler_sequence = 'yxz'
    frameMatchingPointsStatistics:List[FrameMatchingPointsStatistics] = []
    # Initialize detector and the list to store results
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)
    
    number_of_references = len(reference_map)
    for ref_i in range(number_of_references):
        i_frame = reference_map[ref_i]
        ref_i_descriptors = i_frame['descriptors']
        ref_i_keypoints = i_frame['keypoints']
        telemetry_i = i_frame['telemetry']
        ref_hw_i = i_frame['hw']
        i_kps_tensor_gpu = ref_i_keypoints.to(device)
        i_descs_tensor_gpu = ref_i_descriptors.to(device)
        i_hw_tensor = torch.tensor(ref_hw_i, device=device)

        # Create LAFs for the reference keypoints
        i_lafs = KF.laf_from_center_scale_ori(
        i_kps_tensor_gpu[None], torch.ones(1, len(i_kps_tensor_gpu), 1, 1, device=device)
        )
        print('')
        print("Finding matches for frame: ", i_frame['filename'])
        nr_inliers_with_matching_neighbouring_frames:list = []
        for ref_j in range(ref_i+1,number_of_references):
            j_frame = reference_map[ref_j]
            ref_j_descriptors = j_frame['descriptors']
            if i_frame['descriptors'] is None or j_frame['descriptors'] is None:
                continue
            ref_j_keypoints = j_frame['keypoints']
            ref_hw_j = j_frame['hw']
            j_kps_tensor_gpu = ref_j_keypoints.to(device)
            j_descs_tensor_gpu = ref_j_descriptors.to(device)
            j_hw_tensor = torch.tensor(ref_hw_j, device=device)
            j_lafs = KF.laf_from_center_scale_ori(
                j_kps_tensor_gpu[None], torch.ones(1, len(j_kps_tensor_gpu), 1, 1, device=device)
            )

            with torch.inference_mode():
                dists, idxs = lg_matcher(i_descs_tensor_gpu, j_descs_tensor_gpu, 
                                         i_lafs, j_lafs, hw1=i_hw_tensor, hw2=j_hw_tensor)
                

                idxs = idxs.to(torch.device('cpu'))
                valid_rows_mask = (idxs[:, 0] != -1) & (idxs[:, 1] != -1)
                good_idxs = idxs[valid_rows_mask]
                num_current_matches = len(good_idxs)
                if(num_current_matches<min_match_count):
                    continue

                i_keypoints = i_kps_tensor_gpu.to(torch.device('cpu'))
                matched_points_i = i_keypoints[good_idxs[:, 0]]
                matched_points_j = ref_j_keypoints[good_idxs[:, 1]]

                cumulative_score = dists.sum().item()

                H_ij, nr_inliers, i_points_inliers, j_points_inliers = calculate_homography_from_tensors(matched_points_i, 
                                                                                                                        matched_points_j,
                                                                                                                        camera_matrix,
                                                                                                                        ransac_thresh=ransac_threshold,
                                                                                                                        geometry_model=geometry_model)
                if(H_ij is None):
                    continue
                
                nr_inliers_with_matching_neighbouring_frames.append(nr_inliers)

                i_points_inliers = np.squeeze(i_points_inliers)
                j_points_inliers = np.squeeze(j_points_inliers)
                unique_inlier_i_points = np.unique(i_points_inliers, axis=0)
                unique_inlier_j_points = np.unique(j_points_inliers, axis=0)
                nr_unique_inlier_i_points = len(unique_inlier_i_points)
                nr_unique_inlier_j_points = len(unique_inlier_j_points)

                if(nr_inliers>min_match_count and nr_unique_inlier_i_points>min_match_count and nr_unique_inlier_j_points>min_match_count):
                    print("Found match: ", j_frame['filename'])

                    H_ij, _ = cv2.findHomography(i_points_inliers, j_points_inliers, cv2.RANSAC, ransac_threshold)
                    if H_ij is None:
                        print(f"Warning: Could not find Homography for frames {ref_i}-{ref_j}. Skipping pair.")
                        continue

                    telemetry_j = j_frame['telemetry']
                    R_i_world = construct_R_from_telemetry(euler_sequence,
                                telemetry_i['yaw'],
                                telemetry_i['pitch'],
                                telemetry_i['roll']
                            )
                    R_j_world = construct_R_from_telemetry(euler_sequence,
                                telemetry_j['yaw'],
                                telemetry_j['pitch'],
                                telemetry_j['roll']
                            )
                    R_telemetry_ij = R_j_world @ R_i_world.T
                    R_sensors_ij, angular_distances, z_planes, xy_norms = extract_rotation_positive_z_normal(H_ij, camera_matrix, R_telemetry_ij)

                    for R_sensor_ij in R_sensors_ij:
                        sensorTelemetryPair = SensorTelemetryPair()
                        sensorTelemetryPair.sensor_rotation_matrix = R_sensor_ij
                        sensorTelemetryPair.telemetry_rotation_matrix = R_telemetry_ij
                        sensorTelemetryPairs.append(sensorTelemetryPair)
        if(len(nr_inliers_with_matching_neighbouring_frames)>0):
            frameMatching = FrameMatchingPointsStatistics()
            temp_array = np.array(nr_inliers_with_matching_neighbouring_frames)
            avg_nr_inliers_with_neighbour_frames = np.mean(temp_array)
            max_nr_inliers_with_neighbour_frames = np.max(temp_array)
            frameMatching.filename = i_frame['filename']
            frameMatching.avg_number_inliers = avg_nr_inliers_with_neighbour_frames
            frameMatching.max_number_inliers = max_nr_inliers_with_neighbour_frames
            frameMatchingPointsStatistics.append(frameMatching)
        print(f"Average nr inliers with matching neighbouring frames: {avg_nr_inliers_with_neighbour_frames:.1f}")
        print(f"Max nr inliers with matching neighbouring frames: {max_nr_inliers_with_neighbour_frames:.1f}")
    print("Number of matches found: ", len(sensorTelemetryPairs))        

    return sensorTelemetryPairs, frameMatchingPointsStatistics

# used
def cost_function(r_align_rotvec: np.ndarray, pairs: List[SensorTelemetryPair]) -> float:
    """
    Calculates the total squared angular error for a given R_align candidate.
    This is the function we want to minimize.

    Args:
        r_align_rotvec (np.ndarray): A 3-element rotation vector representing R_align.
                                     This is the variable being optimized.
        pairs (List[SensorTelemetryPair]): The list of all (R_sensor, R_telemetry) pairs.

    Returns:
        float: The total squared angular error.
    """
    # 1. Convert the optimization variable (rotation vector) into a matrix
    R_align = ScipyRotation.from_rotvec(r_align_rotvec).as_matrix()
    R_align_T = R_align.T
    
    total_squared_error = 0.0

    for pair in pairs:
        R_s = pair.sensor_rotation_matrix
        R_t = pair.telemetry_rotation_matrix

        R_s_predicted = R_align @ R_t @ R_align_T
        delta_R = R_s_predicted @ R_s.T

        trace = np.trace(delta_R)
        arg = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_error_rad = np.arccos(arg)

        total_squared_error += angle_error_rad**2
    
    return total_squared_error

# used
def find_R_align(pairs: List[SensorTelemetryPair]) -> np.ndarray:
    """
    Finds the optimal R_align matrix by minimizing the cost function.

    Args:
        pairs (List[SensorTelemetryPair]): The list of all (R_sensor, R_telemetry) pairs.

    Returns:
        np.ndarray: The optimized 3x3 R_align rotation matrix.
    """
    if not pairs:
        print("Error: No sensor-telemetry pairs provided.")
        return np.eye(3)

    print(f"Starting optimization with {len(pairs)} pairs...")

    initial_guess_rotvec = np.array([0.0, 0.0, 0.0])

    # Perform the optimization
    result = minimize(
        fun=cost_function,      # The function to minimize
        x0=initial_guess_rotvec,# The initial guess
        args=(pairs,),          # Extra arguments to pass to the cost function
        method='L-BFGS-B'       # A good, general-purpose optimization algorithm
    )

    if result.success:
        print("Optimization successful!")
        # Extract the optimized rotation vector and convert it back to a matrix
        optimized_rotvec = result.x
        R_align_optimized = ScipyRotation.from_rotvec(optimized_rotvec).as_matrix()
        return R_align_optimized
    else:
        print("Error: Optimization failed.")
        print(result.message)
        return np.eye(3) # Return identity as a fallback

class SensorTelemetryPair:
    def __init__(self):
        self.sensor_rotation_matrix = np.eye(3)
        self.telemetry_rotation_matrix = np.eye(3)
        self.sensor_translation_matrix = np.zeros((3, 1))
        self.telemetry_translation_matrix = np.zeros((3, 1))

# used
def evaluate_R_align(R_align:np.ndarray, pairs:SensorTelemetryPair):
    print("\n--- Evaluating R_align Performance ---")
    errors_before = []
    errors_after = []
    R_align_T = R_align.T

    for pair in pairs:
        R_s = pair.sensor_rotation_matrix
        R_t = pair.telemetry_rotation_matrix

        # Error before alignment (R_s vs R_t)
        angle_before = calculate_angular_dist(R_s, R_t)
        errors_before.append(angle_before)

        # Error after alignment (R_s vs predicted R_s)
        R_s_predicted = R_align @ R_t @ R_align_T

        angle_after = calculate_angular_dist(R_s, R_s_predicted)
        errors_after.append(angle_after)

    print(f"Average Angular Error BEFORE alignment: {np.mean(errors_before):.2f} degrees")
    print(f"Average Angular Error AFTER alignment:  {np.mean(errors_after):.2f} degrees")

# used
def make_mask(hw:List[int], roi:List[float])->np.ndarray:
    mask = np.zeros((hw[0], hw[1]), dtype=np.uint8)
    roi_start_row = int(hw[0] * roi[0])
    roi_end_row = int(hw[0] * (1 - roi[1]))
    roi_start_col = int(hw[1] * roi[2])
    roi_end_col = int(hw[1] * (1 - roi[3]))
    mask[roi_start_row:roi_end_row, roi_start_col:roi_end_col] = 255
    return mask

def calculate_angles_from_pixel(pixel_point: np.ndarray, cam_matrix: np.ndarray) -> tuple:
    """
    Calculates the yaw and pitch angles for a single 2D pixel point relative
    to the camera's forward axis.

    This function correctly uses the camera intrinsics and atan2.

    Args:
        pixel_point (np.ndarray): A 1D NumPy array [u, v] for the pixel coordinate.
        cam_matrix (np.ndarray): The 3x3 camera intrinsic matrix.

    Returns:
        tuple: A tuple containing (yaw_degrees, pitch_degrees).
    """
    
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]

    u, v = pixel_point

    u_dif = u - cx
    v_dif = v - cy
    
    yaw_radians = np.arctan2(u_dif, fx)
    
    pitch_radians = np.arctan2(-v_dif, fy)

    yaw_degrees = np.degrees(yaw_radians)
    pitch_degrees = np.degrees(pitch_radians)

    return yaw_degrees, pitch_degrees

# used
def calculate_j_angles(R_sensor_ij, telemetry_i, euler_sequence):
    R_i_from_telemetry = construct_R_from_telemetry(euler_sequence,
        telemetry_i['yaw'], telemetry_i['pitch'], telemetry_i['roll']
    )

    R_j_from_measurement = R_sensor_ij @ R_i_from_telemetry

    frame_j_angles = rotation_matrix_to_euler(euler_sequence, R_j_from_measurement)
    
    return frame_j_angles


#used
def calculate_pose_offset(R_sensor_ij, telemetry_i, telemetry_j, euler_sequence):
    """
    Calculates the angular offset between a measured pose and an expected pose.

    Args:
        R_sensor_ij (np.ndarray): The measured relative rotation from homography.
        telemetry_i (dict): Telemetry angles for the starting frame.
        telemetry_j (dict): Telemetry angles for the ending frame.

    Returns:
        float: The single angular offset in degrees.
    """
        
    frame_j_angles = calculate_j_angles(R_sensor_ij, telemetry_i, euler_sequence)

    yaw_offset = frame_j_angles[0] - telemetry_j['yaw']
    pitch_offset = frame_j_angles[1] - telemetry_j['pitch']
    roll_offset = frame_j_angles[2] - telemetry_j['roll']

    return yaw_offset, pitch_offset, roll_offset

def calculate_angular_error(R_align, pair):
    """Calculates the angular error for a single pair given an R_align."""
    R_s = pair.sensor_rotation_matrix
    R_t = pair.telemetry_rotation_matrix
    
    R_align_T = R_align.T
    R_s_predicted = R_align @ R_t @ R_align_T

    delta_R = R_s_predicted @ R_s.T
    trace = np.trace(delta_R)
    arg = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(arg)

def run_ransac_for_alignment(all_pairs, n=4, k=100, t_deg=2.5):
    """
    Performs RANSAC to find a robust R_align.
    """
    best_inlier_count = -1
    best_model_inliers = []
    best_model_inliers_indices = []
    t_rad = np.deg2rad(t_deg)
    
    for i in range(k):
        # 1. Select a random subset
        random_sample = random.sample(all_pairs, n)
        
        # 2. Fit a model to the subset
        initial_guess_rotvec = np.zeros(3)
        result = minimize(
            cost_function,
            initial_guess_rotvec,
            args=(random_sample,), # Optimize using ONLY the small sample
            method='L-BFGS-B'
        )
        
        if not result.success:
            continue
            
        R_align_candidate = ScipyRotation.from_rotvec(result.x).as_matrix()
        
        # 3. Count inliers from the full dataset
        current_inliers = []
        current_inliers_indices = []
        for indexPair, pair in enumerate(all_pairs):
            error = calculate_angular_error(R_align_candidate, pair)
            if error < t_rad:
                current_inliers.append(pair)
                current_inliers_indices.append(indexPair)
        
        # 4. Update if this is the best model so far
        if len(current_inliers) > best_inlier_count:
            best_inlier_count = len(current_inliers)
            best_model_inliers = current_inliers
            best_model_inliers_indices = current_inliers_indices
            print(f"RANSAC Iteration {i}: Found new best model with {best_inlier_count}/{len(all_pairs)} inliers.")

    # 5. Final refinement using the best inlier set
    if not best_model_inliers:
        print("RANSAC failed to find any consistent model.")
        return None, [], []

    print(f"\nRefining model using the best {len(best_model_inliers)} inliers...")
    final_result = minimize(
        cost_function,
        np.zeros(3),
        args=(best_model_inliers,),
        method='L-BFGS-B'
    )
    
    final_R_align = ScipyRotation.from_rotvec(final_result.x).as_matrix()
    return final_R_align, best_model_inliers, best_model_inliers_indices

def rescale_intrinsic_camera_properties(K, old_width, old_height, new_width, new_height):
    scale_x = new_width / float(old_width)
    scale_y = new_height / float(old_height)

    camMatrix_new = K.copy()
    camMatrix_new[0, 0] *= scale_x  # fx
    camMatrix_new[1, 1] *= scale_y  # fy
    camMatrix_new[0, 2] *= scale_x  # cx
    camMatrix_new[1, 2] *= scale_y  # cy
    return camMatrix_new


def pad_images_in_directory(input_dir, output_dir, target_height=1520):
    """
    Pads all images in an input directory to a target height by adding black
    pixels to the top and saves them to an output directory.

    Args:
        input_dir (str): Path to the directory containing the original images.
        output_dir (str): Path to the directory where padded images will be saved.
        target_height (int): The desired final height of the images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")

    for filename in tqdm(image_files, desc="Padding images"):
        # Construct full file paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            continue

        current_height, current_width = img.shape[:2]

        # Check if padding is needed
        if current_height >= target_height:
            print(f"Image {filename} is already at or above target height. Copying as is.")
            cv2.imwrite(output_path, img)
            continue
            
        # Calculate the padding needed for the top
        padding_top = target_height - current_height
        padding_bottom = 0
        padding_left = 0
        padding_right = 0

        # Add the black border to the top of the image
        # The 'value' argument specifies the color of the border (BGR format)
        padded_img = cv2.copyMakeBorder(
            img,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black color
        )

        # Save the new padded image
        cv2.imwrite(output_path, padded_img)

    print("Processing complete.")

def calculate_new_dimensions(old_height: int, old_width: int, max_long_edge: int) -> Tuple[int, int]:
    """
    Helper function to calculate new dimensions, preserving aspect ratio.
    The longest edge of the new dimensions will be `max_long_edge`.
    """
    longest_edge = max(old_height, old_width)
    scale = max_long_edge / longest_edge
    new_height = int(old_height * scale)
    new_width = int(old_width * scale)
    return new_height, new_width

def resize_images_in_folder(input_path: str, max_dimension: int, output_path: str):
    """
    Resizes all images in an input directory and saves them to an output directory.

    The output directory is first deleted if it exists, then recreated to ensure
    it is empty before processing begins.

    Args:
        input_path (str): The path to the folder containing original images.
        max_dimension (int): The target size for the longest dimension of the images.
        output_path (str): The path to the folder where resized images will be saved.
    """
    # --- 1. Handle the output directory ---
    if os.path.exists(output_path):
        print(f"Output directory '{output_path}' already exists. Deleting it.")
        try:
            shutil.rmtree(output_path)
        except OSError as e:
            print(f"Error deleting directory {output_path}: {e}")
            return

    try:
        os.makedirs(output_path)
        print(f"Successfully created new output directory: '{output_path}'")
    except OSError as e:
        print(f"Error creating directory {output_path}: {e}")
        return

    # --- 2. Find all image files ---
    if not os.path.isdir(input_path):
        print(f"Error: Input directory '{input_path}' not found.")
        return
        
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No image files found in '{input_path}'.")
        return

    print(f"Found {len(image_files)} images to resize to max dimension {max_dimension}.")

    # --- 3. Loop through images, resize, and save ---
    for filename in tqdm(image_files, desc="Resizing images"):
        input_file_path = os.path.join(input_path, filename)
        output_file_path = os.path.join(output_path, filename)

        # Read the image
        image = cv2.imread(input_file_path)
        if image is None:
            print(f"Warning: Could not read image '{input_file_path}'. Skipping.")
            continue

        # Get original dimensions
        old_height, old_width = image.shape[:2]

        # Calculate new dimensions
        new_height, new_width = calculate_new_dimensions(old_height, old_width, max_dimension)

        # Resize the image. INTER_AREA is recommended for down-sampling.
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Save the resized image
        cv2.imwrite(output_file_path, resized_image)

    print("\nProcessing complete. All images have been resized and saved.")
    

def recreate_directory(path: str) -> bool:
    """
    Deletes a directory and all its contents if it exists, then creates a new,
    empty directory in its place.

    Args:
        path (str): The full path of the directory to recreate.

    Returns:
        bool: True if the directory was successfully recreated, False otherwise.
    """
    print(f"--- Preparing to recreate directory: '{path}' ---")
    
    try:
        # 1. Check if the path exists
        if os.path.exists(path):
            # Check if it's actually a directory
            if not os.path.isdir(path):
                print(f"Error: The specified path '{path}' exists but is a file, not a directory.")
                return False
                
            # If it is a directory, delete it and all its contents
            print(f"Directory found. Deleting it completely...")
            shutil.rmtree(path)
            print("Directory successfully deleted.")
        else:
            print("Directory does not exist. No deletion needed.")

        # 2. Create the new, empty directory
        print("Creating new empty directory...")
        os.makedirs(path)
        print(f"Successfully created directory: '{path}'")
        
        return True

    except OSError as e:
        # Catch potential permission errors or other OS-level issues
        print(f"\nAn OS error occurred: {e}")
        print("Please check your permissions for the target location.")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\nAn unexpected error occurred: {e}")
        return False


def print_matching_statistics_summary(stats_list: List[FrameMatchingPointsStatistics]):
    """
    Analyzes a list of FrameMatchingPointsStatistics and prints a summary
    of the best-performing frames.
    """
    if not stats_list:
        print("No matching statistics to analyze.")
        return

    # Find the frame with the highest average number of inliers
    best_avg = max(stats_list, key=lambda stat: stat.avg_number_inliers)

    # Find the frame with the highest maximum number of inliers
    best_max = max(stats_list, key=lambda stat: stat.max_number_inliers)

    print("-" * 50)
    print("Feature Matching Performance Summary")
    print("-" * 50)
    
    print(f"Highest Average Inliers:")
    print(f"  - Filename: {best_avg.filename}")
    print(f"  - Average Inlier Count: {best_avg.avg_number_inliers:.2f}")
    
    print("\nHighest Maximum Inliers:")
    print(f"  - Filename: {best_max.filename}")
    print(f"  - Maximum Inlier Count: {best_max.max_number_inliers}")
    
    print("-" * 50)
