import os
import torch
import kornia.feature as KF
import cv2
import numpy as np

from .utils import save_reference_map_to_COLMAP_txts

def append_matches_to_file(
    match_idxs: torch.Tensor,
    image1_name: str,
    image2_name: str,
    output_filepath: str
):
    """
    Appends bidirectional matches to a COLMAP-style matches text file.

    Args:
        match_idxs (torch.Tensor): A tensor of shape (N, 2) where N is the
            number of matches. Each row contains [index_in_image1, index_in_image2].
        image1_name (str): The filename of the first image.
        image2_name (str): The filename of the second image.
        output_filepath (str): The path to the text file to append to.
    """
    # Check if there are any matches to write
    if len(match_idxs) == 0:
        print(f"No matches to write for pair ({image1_name}, {image2_name}).")
        return

    # For efficiency, it's faster to convert to a NumPy array or list on the CPU
    # before iterating for string formatting.
    matches_list = match_idxs.cpu().tolist()

    # --- Prepare the forward matches (Image 1 -> Image 2) ---
    forward_header = f"{image1_name} {image2_name}"
    # Use a list comprehension to create all match strings
    forward_matches_str = "\n".join([f"{int(m[0])} {int(m[1])}" for m in matches_list])

    # --- Prepare the reverse matches (Image 2 -> Image 1) ---
    reverse_header = f"{image2_name} {image1_name}"
    reverse_matches_str = "\n".join([f"{int(m[1])} {int(m[0])}" for m in matches_list])

    # Combine everything into a single string to be written
    # Add newlines for spacing between blocks, which is good practice
    output_string = (
        f"{forward_header}\n"
        f"{forward_matches_str}\n\n"
        f"{reverse_header}\n"
        f"{reverse_matches_str}\n\n"
    )

    # Open the file in append mode ('a') and write the entire block
    with open(output_filepath, 'a') as f:
        f.write(output_string)
        
    print(f"Successfully appended {len(matches_list)} matches for pair ({image1_name}, {image2_name}) to {output_filepath}")

def parse_colmap_txt(file_path: str):
    """
    Parses a COLMAP-style feature file and extracts keypoints and descriptors
    into PyTorch tensors.

    Args:
        file_path (str or io.StringIO): The path to the .txt feature file or
                                        a file-like object.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - keypoints (torch.Tensor): A tensor of shape (N, 4) where N is the
              number of keypoints. Each row is [x, y, scale, orientation].
            - descriptors (torch.Tensor): A tensor of shape (N, D) where D is
              the descriptor dimension (e.g., 128).
    """
    
    # Use 'with open' for safe file handling
    if isinstance(file_path, str):
        f = open(file_path, 'r')
    else:
        # Allows passing a file-like object like io.StringIO for testing
        f = file_path

    with f:
        # Read the header line to get number of keypoints and descriptor dimension
        header = f.readline().strip().split()
        num_keypoints, descriptor_dim = int(header[0]), int(header[1])

        # Pre-allocate lists for efficiency
        keypoints_list = []
        descriptors_list = []

        # Loop through the rest of the lines
        for line in f:
            # Split the line into numbers, convert them to floats
            # Using a list comprehension is faster than a for loop here
            data = [float(x) for x in line.strip().split()]
            
            # The first 4 numbers are keypoint data (x, y, scale, orientation)
            keypoint_data = data[:2]
            keypoints_list.append(keypoint_data)

            # The rest of the numbers are the descriptor
            descriptor_data = data[4:]
            descriptors_list.append(descriptor_data)

    # Convert the Python lists to PyTorch Tensors
    # Specify dtype=torch.float32 for consistency with ML models
    keypoints = torch.tensor(keypoints_list, dtype=torch.float32)
    descriptors = torch.tensor(descriptors_list, dtype=torch.float32)
    
    # Sanity check: ensure the shapes match the header
    assert keypoints.shape == (num_keypoints, 2)
    assert descriptors.shape == (num_keypoints, descriptor_dim)

    return keypoints, descriptors

def process_files_and_give_me_desc_kp_lafs(colmap_output, 
                                           colmap_files, 
                                           file_index, 
                                           device):
    file_name = colmap_files[file_index]
    colmap_file = os.path.join(colmap_output, file_name)
    keypoints, descriptors = parse_colmap_txt(colmap_file)
    keypoints = keypoints.to(device)
    descriptors = descriptors.to(device)
    lafs = KF.laf_from_center_scale_ori(keypoints[None], torch.ones(1, len(keypoints), 1, 1, device=device))
    return descriptors, keypoints, lafs

def main():
    REFERENCE_PANORAMA_PATH = './data/referencePanoramaMultiZoom.json'
    colmap_output = './data/colmap_multizoom/'
    save_reference_map_to_COLMAP_txts(REFERENCE_PANORAMA_PATH, colmap_output)
    
    match_pairs_file = os.path.join(colmap_output, 'pair_matches.txt')
    image_height = 1520
    image_width = 2688
    min_match_count = 100
    ransac_thresh = 2.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)

    colmap_files = os.listdir(colmap_output)
    nr_of_files = len(colmap_files)
    for i in range(nr_of_files):
        i_file_name = colmap_files[i]
        if(not(i_file_name.endswith('.txt'))):
            continue
        print("===========================")
        print("=============" + str(i) + "============")
        print("===========================")
        i_descriptors, i_keypoints, i_lafs = process_files_and_give_me_desc_kp_lafs(colmap_output, 
                                           colmap_files, 
                                           i, 
                                           device)
        i_hw_tensor = torch.tensor([image_height, image_width], device=device)

        for j in range(i+1,nr_of_files):
            j_file_name = colmap_files[j]
            if(not(j_file_name.endswith('.txt'))):
                continue
            j_descriptors, j_keypoints, j_lafs = process_files_and_give_me_desc_kp_lafs(colmap_output, 
                                           colmap_files, 
                                           j, 
                                           device)
            j_hw_tensor = torch.tensor([image_height, image_width], device=device)
            with torch.inference_mode():
                dists, idxs = lg_matcher(i_descriptors, j_descriptors, i_lafs, j_lafs, hw1=i_hw_tensor, hw2=j_hw_tensor)
                idxs = idxs.to(torch.device('cpu'))
                valid_rows_mask = (idxs[:, 0] != -1) & (idxs[:, 1] != -1)
                good_idxs = idxs[valid_rows_mask]
                num_current_matches = len(good_idxs)
                if(num_current_matches<min_match_count):
                    continue

                i_keypoints = i_keypoints.to(torch.device('cpu'))
                matched_points_i = i_keypoints[good_idxs[:, 0]]
                matched_points_j = j_keypoints[good_idxs[:, 1]]

                i_pts_np = matched_points_i.cpu().numpy()
                j_pts_np = matched_points_j.cpu().numpy()

                homography_matrix, mask = cv2.findHomography(
                            i_pts_np, 
                            j_pts_np, 
                            cv2.RANSAC, 
                            ransac_thresh
                        )
                
                if homography_matrix is None:
                    print("Warning: Homography calculation failed. The points might be collinear.")
                    continue

                inlier_mask = mask.ravel() == 1
                num_inliers = np.sum(mask)
                if(num_inliers<min_match_count):
                    print("Match rejected due to low number of inliers")
                    continue

                inlier_matching_idxs = good_idxs[inlier_mask]

                append_matches_to_file(inlier_matching_idxs, 
                                       i_file_name,
                                       j_file_name,
                                       match_pairs_file
                                       )
   
                
            
        



if __name__ == '__main__':
    main()