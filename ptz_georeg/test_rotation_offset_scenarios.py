from scipy.spatial.transform import Rotation as ScipyRotation

from utils import construct_R_from_telemetry, rotation_matrix_to_euler, normalize_angle_delta

if __name__ == '__main__':
    euler_sequence = 'yxz'
    yaw_offset = 10
    pitch_offset_telemetry = 14
    pitch_value = 88
    telemetry_i = {'yaw': 40, 'pitch': pitch_value, 'roll': 0}
    telemetry_j = {'yaw': 40, 'pitch': pitch_value + pitch_offset_telemetry, 'roll': 0}
    real_i = {'yaw': 40, 'pitch': pitch_value, 'roll': 0.0}
    real_j = {'yaw': 40 + yaw_offset, 'pitch': pitch_value, 'roll': 1.5}

    R_telemetry_i = construct_R_from_telemetry(euler_sequence, telemetry_i['yaw'], telemetry_i['pitch'], telemetry_i['roll'])
    R_telemetry_j = construct_R_from_telemetry(euler_sequence, telemetry_j['yaw'], telemetry_j['pitch'], telemetry_j['roll'])

    R_real_i = construct_R_from_telemetry(euler_sequence, real_i['yaw'], real_i['pitch'], real_i['roll'])
    R_real_j = construct_R_from_telemetry(euler_sequence, real_j['yaw'], real_j['pitch'], real_j['roll'])

    R_telemetry_ij = R_telemetry_j @ R_telemetry_i.T
    R_real_ij = R_real_j @ R_real_i.T

    R_real_j_est = R_real_ij @ R_telemetry_i

    R_diff = R_real_j_est @ R_telemetry_j.T

    diff_yaw, diff_pitch, diff_roll = rotation_matrix_to_euler(euler_sequence, R_diff)

    yaw_j_est, pitch_j_est, roll_j_est = rotation_matrix_to_euler(euler_sequence, R_real_j_est)
    yaw_offset_angle_difference = normalize_angle_delta(yaw_j_est - telemetry_j['yaw'])
    pitch_offset_angle_difference = normalize_angle_delta(pitch_j_est - telemetry_j['pitch'])
    roll_offset_angle_difference = normalize_angle_delta(roll_j_est - telemetry_j['roll'])

    print("The Euler angles of the rotation that transforms telemetry_j to real_j are:")
    print(f"Yaw:   {diff_yaw:.4f}")
    print(f"Pitch: {diff_pitch:.4f}")
    print(f"Roll:  {diff_roll:.4f}")
    

    print("The Euler Angle Differences:")
    print(f"Yaw:   {yaw_offset_angle_difference:.4f}")
    print(f"Pitch: {pitch_offset_angle_difference:.4f}")
    print(f"Roll:  {roll_offset_angle_difference:.4f}")
    


    