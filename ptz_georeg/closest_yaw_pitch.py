import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

def extract_closest_yaw_pitch_roll(R_sensor: np.ndarray, yaw_telemetry: float, pitch_telemetry: float, roll_telemetry: float) -> tuple:
    """
    Finds the yaw and pitch angles that create a no-roll rotation matrix
    closest to the measured R_sensor.

    This function is designed to measure the "effective" yaw and pitch from a full
    3D rotation, constrained by the knowledge that the source (a PTZ) has no roll.

    Args:
        R_sensor (np.ndarray): The 3x3 rotation matrix from measurement (e.g., homography).
        yaw_telemetry (float): The telemetry yaw in degrees (used as a good initial guess).
        pitch_telemetry (float): The telemetry pitch in degrees (used as a good initial guess).

    Returns:
        tuple: A tuple containing:
            - yaw_measured (float): The optimized yaw angle in degrees.
            - pitch_measured (float): The optimized pitch angle in degrees.
    """

    def objective_function(angles_deg):
        """
        This is the function we want to minimize.
        It calculates the angular distance between R_sensor and a matrix
        constructed from the given yaw/pitch angles.
        """
        yaw_deg, pitch_deg, roll_deg = angles_deg

        # Construct the ideal no-roll PTZ rotation: yaw followed by pitch
        r_yaw = Rotation.from_euler('y', yaw_deg, degrees=True)
        r_pitch = Rotation.from_euler('x', pitch_deg, degrees=True)
        r_roll = Rotation.from_euler('z', roll_deg, degrees=True)
        R_candidate = (r_yaw * r_pitch * r_roll).as_matrix()

        # Calculate the "error" rotation
        R_error = R_sensor @ R_candidate.T
        
        # Calculate the angular distance (our cost)
        trace = np.trace(R_error)
        angle_rad = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))
        
        return angle_rad

    # The telemetry angles are the perfect starting point for the search
    initial_guess = [yaw_telemetry, pitch_telemetry, roll_telemetry]

    # Run the optimization
    result = minimize(
        objective_function,
        initial_guess,
        method='Nelder-Mead' # A good, simple optimizer for this problem
    )

    # The result contains the best-fit yaw and pitch
    yaw_measured, pitch_measured, roll_measured = result.x
    
    return yaw_measured, pitch_measured, roll_measured


def extract_closest_yaw_pitch(R_sensor: np.ndarray, yaw_telemetry: float, pitch_telemetry: float) -> tuple:
    """
    Finds the yaw and pitch angles that create a no-roll rotation matrix
    closest to the measured R_sensor.

    This function is designed to measure the "effective" yaw and pitch from a full
    3D rotation, constrained by the knowledge that the source (a PTZ) has no roll.

    Args:
        R_sensor (np.ndarray): The 3x3 rotation matrix from measurement (e.g., homography).
        yaw_telemetry (float): The telemetry yaw in degrees (used as a good initial guess).
        pitch_telemetry (float): The telemetry pitch in degrees (used as a good initial guess).

    Returns:
        tuple: A tuple containing:
            - yaw_measured (float): The optimized yaw angle in degrees.
            - pitch_measured (float): The optimized pitch angle in degrees.
    """

    def objective_function(angles_deg):
        """
        This is the function we want to minimize.
        It calculates the angular distance between R_sensor and a matrix
        constructed from the given yaw/pitch angles.
        """
        yaw_deg, pitch_deg = angles_deg

        # Construct the ideal no-roll PTZ rotation: yaw followed by pitch
        r_yaw = Rotation.from_euler('y', yaw_deg, degrees=True)
        r_pitch = Rotation.from_euler('x', pitch_deg, degrees=True)
        R_candidate = (r_yaw * r_pitch).as_matrix()

        # Calculate the "error" rotation
        R_error = R_sensor @ R_candidate.T
        
        # Calculate the angular distance (our cost)
        trace = np.trace(R_error)
        angle_rad = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))
        
        return angle_rad

    # The telemetry angles are the perfect starting point for the search
    initial_guess = [yaw_telemetry, pitch_telemetry]

    # Run the optimization
    result = minimize(
        objective_function,
        initial_guess,
        method='Nelder-Mead' # A good, simple optimizer for this problem
    )

    # The result contains the best-fit yaw and pitch
    yaw_measured, pitch_measured = result.x
    
    return yaw_measured, pitch_measured

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == '__main__':
    # --- 1. Define the Scenario ---
    # The known telemetry command
    yaw_tel = 40.0
    pitch_tel = 10.0

    # Let's simulate a physical misalignment (the unknown R_align)
    # e.g., the camera is rolled 5 degrees and pitched down 2 degrees on its mount
    R_align = (Rotation.from_euler('x', 5, degrees=True) * Rotation.from_euler('y', -2, degrees=True)).as_matrix()

    # --- 2. Simulate the "True" Rotation ---
    # First, build the ideal telemetry rotation (how the base moves)
    r_yaw_tel = Rotation.from_euler('y', yaw_tel, degrees=True)
    r_pitch_tel = Rotation.from_euler('x', pitch_tel, degrees=True)
    R_telemetry = (r_yaw_tel * r_pitch_tel).as_matrix()

    # Now, calculate what the sensor *actually* sees because of the misalignment
    # This is the matrix your homography decomposition would give you.
    R_sensor_measured = R_align.T @ R_telemetry @ R_align
    
    print("--- Inputs ---")
    print(f"Telemetry Command: Yaw={yaw_tel:.2f}°, Pitch={pitch_tel:.2f}°")
    print("Measured R_sensor (from homography):\n", np.round(R_sensor_measured, 3))
    
    # Let's see what a naive decomposition gives (it will be wrong)
    naive_angles = Rotation.from_matrix(R_sensor_measured).as_euler('yxz', degrees=True)
    print(f"\nNaive Decomposition (YXZ): Yaw={naive_angles[0]:.2f}, Pitch={naive_angles[1]:.2f}, Roll={naive_angles[2]:.2f}°")
    print("Notice how the yaw and pitch are not close to telemetry, and roll is non-zero.")


    # --- 3. Use the Correct Method to Extract the Closest Angles ---
    yaw_meas, pitch_meas = extract_closest_yaw_pitch(R_sensor_measured, yaw_tel, pitch_tel)

    print("\n" + "="*50)
    print("--- Results from Optimization ---")
    print(f"Extracted Measured Yaw:   {yaw_meas:.2f}°")
    print(f"Extracted Measured Pitch: {pitch_meas:.2f}°")

    # --- 4. Calculate the Offsets You Wanted ---
    yaw_offset = yaw_meas - yaw_tel
    pitch_offset = pitch_meas - pitch_tel

    print("\n--- Final Offsets ---")
    print(f"Yaw Offset:   {yaw_offset:.2f}°")
    print(f"Pitch Offset: {pitch_offset:.2f}°")