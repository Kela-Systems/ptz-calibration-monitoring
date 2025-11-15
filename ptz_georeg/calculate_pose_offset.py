
from .utils import construct_R_from_telemetry, calculate_pose_offset

if __name__ == '__main__':
    # --- EXAMPLE ---
    euler_sequence = 'yxz'
    yaw_offset = 2
    pitch_offset = 1.5
    real_i = {'yaw': 40, 'pitch': 10, 'roll': 0}
    real_j = {'yaw': 60 + yaw_offset, 'pitch': 25 + pitch_offset, 'roll': 0}
    R_real_i = construct_R_from_telemetry(euler_sequence, real_i['yaw'], real_i['pitch'], real_i['roll'])
    R_real_j = construct_R_from_telemetry(euler_sequence, real_j['yaw'], real_j['pitch'], real_j['roll'])
    R_sensor_ij = R_real_j @ R_real_i.T

    telemetry_i = {'yaw': 40, 'pitch': 10, 'roll': 0}
    telemetry_j = {'yaw': 60, 'pitch': 25, 'roll': 0} # 20 deg yaw, 15 deg pitch delta

    offsets = calculate_pose_offset(R_sensor_ij, telemetry_i, telemetry_j, euler_sequence)

    print(offsets)
    #print(f"The total angular offsets between the measured pose and the expected telemetry pose are: {offsets:.2f} degrees")
    # The result should be very close to our simulated 1.0 degree error.