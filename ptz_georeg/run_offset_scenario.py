from scipy.spatial.transform import Rotation as ScipyRotation

def rotation_matrix_to_euler_this(telemetry_sequence,R):
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

def construct_R_from_telemetry_this(euler_sequence, pan_deg, pitch_deg, roll_deg):
    """
    Constructs an absolute rotation matrix from telemetry data.
    
    Args:
        pan_deg (float): Yaw angle from telemetry. Positive is "turning right".
        pitch_deg (float): Pitch angle from telemetry. Positive is "looking up".
        roll_deg (float): Roll angle from telemetry.
        
    Returns:
        np.ndarray: A 3x3 rotation matrix representing the camera's orientation.
    """
    yaw_corrected = - pan_deg 
    pitch_corrected = - pitch_deg
    roll_corrected = roll_deg
    
    rotation = ScipyRotation.from_euler(
        euler_sequence, 
        [yaw_corrected, pitch_corrected, roll_corrected], 
        degrees=True
    )
    
    return rotation.as_matrix()

def main():
    euler_sequence = 'yxz'
    yaw_ref = 180
    pitch_ref = 10
    roll_ref = 0.0

    R_ref = construct_R_from_telemetry_this(euler_sequence,
                               yaw_ref,
                               pitch_ref,
                               roll_ref
                               )
    
    yaw_offset = 0
    pitch_offset = 5
    roll_offset = 0
    
    R_offset = construct_R_from_telemetry_this(euler_sequence,
                                               yaw_offset,
                                               pitch_offset,
                                               roll_offset)
    
    R_combined = R_ref @ R_offset

    angles = rotation_matrix_to_euler_this(euler_sequence, R_combined)
    print(angles)

    yaw_query = 180
    pitch_query = 10
    roll_query = 0.0

    R_query = construct_R_from_telemetry_this(euler_sequence,
                                                  yaw_query,
                                                  pitch_query,
                                                  roll_query 
                                                  )
    
    R_telemetry_query_ref = R_ref@R_query.T
    angles = rotation_matrix_to_euler_this(euler_sequence, R_telemetry_query_ref)
    print(angles)

    
    R_query_visual = R_query @ R_offset
    R_visual_query_ref = R_ref@R_query_visual.T

    R_offset_predicted = R_query.T @ R_visual_query_ref.T @ R_ref

    angles = rotation_matrix_to_euler_this(euler_sequence, R_offset_predicted)
    print(angles)




if __name__ == '__main__':
    main()