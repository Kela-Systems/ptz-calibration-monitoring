import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

def find_align_to_zero_roll(R_phys, R_tel):
    """
    Finds the R_align that minimizes the roll component of the final offset.
    """

    def cost_function(r_align_rotvec):
        """
        The cost is the absolute value of the roll angle of the corrected offset.
        """
        # 1. Construct the candidate R_align
        R_align = Rotation.from_rotvec(r_align_rotvec).as_matrix()
        
        # 2. Calculate the corrected visual measurement
        R_visual_corrected = R_align @ R_phys @ R_align.T
        
        # 3. Calculate the final offset matrix
        R_offset_change = R_visual_corrected @ R_tel.T
        
        # 4. Decompose the offset and get the roll angle
        try:
            # Use 'yxz' as our standard for Yaw, Pitch, Roll
            _, _, roll_deg = Rotation.from_matrix(R_offset_change).as_euler('yxz', degrees=True)
        except Exception:
            # In case of matrix instability, return a large penalty
            return 1e6
            
        # 5. The cost is the absolute value of the roll. We want this to be zero.
        return abs(roll_deg)

    # Initial guess: a zero rotation
    initial_guess = np.array([0.0, 0.0, 0.0])

    print("--- Searching for R_align that zeros the roll offset... ---")
    
    # Run the optimization
    result = minimize(
        cost_function,
        initial_guess,
        method='L-BFGS-B'
    )

    if result.fun < 1e-5: # Check if roll is successfully minimized to near-zero
        print("Optimization successful!")
        R_align_optimal = Rotation.from_rotvec(result.x).as_matrix()
        return R_align_optimal
    else:
        print("Optimization did not converge to a zero-roll solution.")
        return None

if __name__ == '__main__':
    # --- Your Data ---
    # R_Physical = np.array([
    #     [ 0.9894,  0.0839, -0.1182],
    #     [-0.0825,  0.9965,  0.0165],
    #     [ 0.1191, -0.0065,  0.9929]
    # ])

    # R_telemetry = np.array([
    #     [ 1.    ,  0.0011, -0.0026],
    #     [-0.0016,  0.9724, -0.2334],
    #     [ 0.0023,  0.2334,  0.9724]
    # ])

    # R_Physical = np.array([
    #     [0.9711, 0.2128, 0.108],
    #     [-0.2074, 0.9765, -0.0588],
    #     [-0.118, 0.0347, 0.9924]])
    
    # R_telemetry = np.array([
    #     [ 1., -0.0021, 0.0002],
    #     [0.0021, 0.9724, -0.2334],
    #     [0.0003, 0.2334, 0.9724]])
    
    R_Physical = np.array([
        [0.978, 0.0669, -0.1977],
        [-0.067, 0.9977, 0.006],
        [0.1976, 0.0074, 0.9802]])
    
    R_telemetry = np.array([
        [ 1., 0.0025, -0.006],
        [-0.0038, 0.9724, -0.2334],
        [0.0052, 0.2335, 0.9724]])
    

    # --- Run the Proof ---

    # 1. Analyze the uncalibrated state
    R_offset_uncalibrated = R_Physical @ R_telemetry.T
    yaw_unc, pitch_unc, roll_unc = Rotation.from_matrix(R_offset_uncalibrated).as_euler('yxz', degrees=True)

    print("--- Uncalibrated State ---")
    print(f"Initial Offsets: Yaw={yaw_unc:.2f}°, Pitch={pitch_unc:.2f}°, Roll={roll_unc:.2f}°")
    print(f"(Note the non-zero roll of {roll_unc:.2f}°)")

    # 2. Find the R_align that specifically targets and eliminates the roll
    R_align_solution = find_align_to_zero_roll(R_Physical, R_telemetry)

    if R_align_solution is not None:
        print("\n--- Applying the Solution ---")
        print("Found an R_align that zeros the roll:")
        print(np.round(R_align_solution, 4))
        
        # Decompose this R_align to understand it
        yaw_align, pitch_align, roll_align = Rotation.from_matrix(R_align_solution).as_euler('yxz', degrees=True)
        print(f"\nThis R_align represents a physical misalignment of:")
        print(f"  Yaw={yaw_align:.2f}°, Pitch={pitch_align:.2f}°, Roll={roll_align:.2f}°")
        
        # 3. Verify the result
        R_visual_final = R_align_solution @ R_Physical @ R_align_solution.T
        R_offset_final = R_visual_final @ R_telemetry.T
        
        yaw_final, pitch_final, roll_final = Rotation.from_matrix(R_offset_final).as_euler('yxz', degrees=True)

        print("\n--- Calibrated State ---")
        print("Final Offsets after applying the calculated R_align:")
        print(f"  Final Yaw Offset:   {yaw_final:.2f}°")
        print(f"  Final Pitch Offset: {pitch_final:.2f}°")
        print(f"  Final Roll Offset:  {roll_final:.4f}° (This is now effectively zero)")