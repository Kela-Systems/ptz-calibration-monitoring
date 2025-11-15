import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as ScipyRotation

def euler_to_matrix(yaw, pitch, roll, sequence='yxz'):
    """Helper to create a rotation matrix."""
    r = ScipyRotation.from_euler(sequence, [yaw, pitch, roll], degrees=True)
    return r.as_matrix()

def matrix_to_euler(R, sequence='yxz'):
    """Helper to decompose a rotation matrix."""
    r = ScipyRotation.from_matrix(R)
    return r.as_euler(sequence, degrees=True) # Returns [yaw, pitch, roll]

def simulate_offset_full_orientation(yaw_deg: float, pitch_deg: float, 
                                       trans_yaw_deg: float, trans_pitch_deg: float, trans_roll_deg: float,
                                       euler_sequence='yxz') -> Tuple[float, float, float]:
    """
    Calculates the final world orientation of a crooked camera given a local command.
    This is a more physically complete method that correctly calculates induced roll.

    Args:
        yaw_deg: The yaw command sent to the camera's motors (degrees).
        pitch_deg: The pitch command sent to the camera's motors (degrees).
        trans_yaw_deg: The yaw of the crooked mount (degrees).
        trans_pitch_deg: The pitch of the crooked mount (degrees).
        trans_roll_deg: The roll of the crooked mount (degrees).
        euler_sequence: The Euler sequence to use for all calculations.

    Returns:
        (world_yaw, world_pitch, world_roll): The final orientation in the world frame.
    """
    # 1. Create the rotation matrix for the fixed mount offset.
    R_offset = euler_to_matrix(trans_yaw_deg, trans_pitch_deg, trans_roll_deg, sequence=euler_sequence)
    
    # 2. Create the rotation matrix for the motor command.
    #    A standard PTZ command has no roll, so roll_deg = 0.
    R_command = euler_to_matrix(yaw_deg, pitch_deg, 0, sequence=euler_sequence)
    
    # 3. Chain the rotations to find the final world orientation.
    #    Apply the command first, then the offset.
    R_total_world = R_command @ R_offset
    
    # 4. Decompose the final rotation matrix into Euler angles.
    world_yaw, world_pitch, world_roll = matrix_to_euler(R_total_world, sequence=euler_sequence)
    

    R_applied_world = euler_to_matrix(world_yaw, world_pitch, 0.0)
    R_applied_offset = R_command.T @ R_applied_world

    yaw_offset_applied, pitch_offset_applied, roll_offset_applied = matrix_to_euler(R_applied_offset, sequence=euler_sequence)


    return [world_yaw, world_pitch, world_roll], [yaw_offset_applied, pitch_offset_applied, roll_offset_applied]


def simulate_offset(yaw_deg, pitch_deg, trans_yaw_deg, trans_pitch_deg, trans_roll_deg):
    """
    Calculates the command to send to a PERFECT camera to simulate the behavior
    of a CROOKED camera.

    Args:
        yaw_deg (float): The yaw command you would send to the crooked camera.
        pitch_deg (float): The pitch command you would send to the crooked camera.
        trans_yaw_deg (float): The rotational offset in yaw of the crooked camera.
        trans_pitch_deg (float): The rotational offset in pitch of the crooked camera.
        trans_roll_deg (float): The rotational offset in roll of the crooked camera.

    Returns:
        (yaw_prime_deg, pitch_prime_deg): The new commands to send to your perfect camera.
    """
    # 1. Convert the hypothetical command into a 3D vector in the camera's LOCAL frame.
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    local_direction = np.array([x, y, z])

    # 2. Define the rotation that represents the camera's offset.
    trans_pitch = np.radians(trans_pitch_deg)
    trans_yaw = np.radians(trans_yaw_deg)
    trans_roll = np.radians(trans_roll_deg)
    
    Rz = np.array([
        [np.cos(trans_yaw), -np.sin(trans_yaw), 0],
        [np.sin(trans_yaw),  np.cos(trans_yaw), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(-trans_pitch), 0, np.sin(-trans_pitch)],
        [0, 1, 0],
        [-np.sin(-trans_pitch), 0, np.cos(-trans_pitch)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(trans_roll), -np.sin(trans_roll)],
        [0, np.sin(trans_roll), np.cos(trans_roll)]
    ])

    R = Rz @ Ry @ Rx

    # 3. THE KEY CHANGE: Apply the forward rotation (R, not R.T) to find
    #    where the crooked camera is ACTUALLY pointing in the true world.
    world_direction = R @ local_direction

    # 4. Convert this true world direction back into yaw/pitch commands for your
    #    perfect camera.
    x_world, y_world, z_world = world_direction

    pitch_prime = np.arcsin(np.clip(z_world, -1.0, 1.0))
    yaw_prime = np.arctan2(y_world, x_world)

    return np.degrees(yaw_prime), np.degrees(pitch_prime)


def apply_rotation_to_camera_command(yaw_deg: float, pitch_deg: float, 
                                     trans_pitch_deg: float, trans_yaw_deg: float, trans_roll_deg: float) -> Tuple[float, float]:
    """
    Apply a 3D rotation (representing camera base rotation) to a pan/tilt command.
    
    This simulates a physical rotation of the camera mount. The camera receives commands
    in its rotated reference frame.
    
    Args:
        yaw_deg: Desired yaw/pan in the world frame (degrees)
        pitch_deg: Desired pitch/tilt in the world frame (degrees)
        trans_pitch_deg: Pitch rotation of the camera base (degrees)
        trans_yaw_deg: Yaw rotation of the camera base (degrees)
        trans_roll_deg: Roll rotation of the camera base (degrees)
    
    Returns:
        (rotated_yaw, rotated_pitch): The pan/tilt commands to send to the physically rotated camera
    """
    # Convert degrees to radians
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    trans_pitch = np.radians(trans_pitch_deg)
    trans_yaw = np.radians(trans_yaw_deg)
    trans_roll = np.radians(trans_roll_deg)
    
    # Convert spherical camera pointing direction to a 3D unit vector
    # Standard camera convention: pitch=0, yaw=0 points along +X axis
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    direction = np.array([x, y, z])
    
    # Build rotation matrix for the camera base rotation (ZYX Euler angles)
    # This represents the physical rotation of the camera mount
    # Applied in order: Yaw (Z), Pitch (Y), Roll (X)
    
    # Rotation around Z-axis (yaw)
    Rz = np.array([
        [np.cos(trans_yaw), -np.sin(trans_yaw), 0],
        [np.sin(trans_yaw),  np.cos(trans_yaw), 0],
        [0, 0, 1]
    ])
    
    # Rotation around Y-axis (pitch) - NOTE: Negative sign for pitch to match camera conventions
    # Positive pitch = tilt up, which requires rotating the direction vector down to compensate
    Ry = np.array([
        [np.cos(-trans_pitch), 0, np.sin(-trans_pitch)],
        [0, 1, 0],
        [-np.sin(-trans_pitch), 0, np.cos(-trans_pitch)]
    ])
    
    # Rotation around X-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(trans_roll), -np.sin(trans_roll)],
        [0, np.sin(trans_roll), np.cos(trans_roll)]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx (applied right to left)
    # R represents how the camera base is rotated relative to the world frame
    R = Rz @ Ry @ Rx
    
    # We want to find what command to send to the rotated camera to point in the desired world direction
    # The world direction needs to be expressed in the camera's local (rotated) frame
    # We use the inverse rotation: R.T (or R^-1 for rotation matrices, R.T = R^-1)
    # This transforms FROM world frame TO camera's local frame
    rotated_direction = R.T @ direction
    
    # Convert back to spherical coordinates
    x_rot, y_rot, z_rot = rotated_direction
    
    # Calculate rotated pitch and yaw
    rotated_pitch = np.arcsin(np.clip(z_rot, -1.0, 1.0))
    rotated_yaw = np.arctan2(y_rot, x_rot)
    
    # Convert back to degrees
    return np.degrees(rotated_yaw), np.degrees(rotated_pitch)

def calculate_commands_and_print(command_yaw,
                                 command_pitch,
                                 offset_pitch,
                                 offset_yaw,
                                 offset_roll):
    yaw_prime_deg_one, pitch_prime_deg_one =  apply_rotation_to_camera_command(command_yaw, command_pitch, 
                                     offset_pitch, offset_yaw, offset_roll)

    yaw_prime_deg_two, pitch_prime_deg_two = simulate_offset(
    command_yaw, command_pitch, 
    offset_yaw, offset_pitch, offset_roll)

    simulate_result = simulate_offset_full_orientation(command_yaw, command_pitch, 
                                       offset_yaw, offset_pitch, offset_roll)
    
    yaw_prime_deg_three, pitch_prime_deg_three, roll_prime_deg_three = simulate_result[0]
    
    yaw_applied_offset, pitch_applied_offset, roll_applied_offset = simulate_result[1]
    
    # print("")
    # print("with R.T")
    # print(f"To simulate a camera with a {offset_yaw}-degree yaw offset and a {offset_pitch}-degree pitch offset receiving a ({command_yaw}, {command_pitch}) command...")
    # print(f"You must send this command to your perfect camera: (yaw={yaw_prime_deg_one:.2f}, pitch={pitch_prime_deg_one:.2f})")

    # print("")
    # print("with R")
    # print(f"To simulate a camera with a {offset_yaw}-degree yaw offset and a {offset_pitch}-degree pitch offset receiving a ({command_yaw}, {command_pitch}) command...")
    # print(f"You must send this command to your perfect camera: (yaw={yaw_prime_deg_two:.2f}, pitch={pitch_prime_deg_two:.2f})")
    # print("")

    print("")
    print("with simulate offset full orientation")
    print(f"To simulate a camera with a {offset_yaw}-degree yaw offset and a {offset_pitch}-degree pitch offset receiving a ({command_yaw}, {command_pitch}) command...")
    print(f"You must send this command to your perfect camera: (yaw={yaw_prime_deg_three:.2f}, pitch={pitch_prime_deg_three:.2f}, roll={roll_prime_deg_three})")
    print("")
    print(f"By sending commands: (yaw={yaw_prime_deg_three:.2f}, pitch={pitch_prime_deg_three:.2f}, roll={0.0})")
    print(f"The applied offset is: (yaw={yaw_applied_offset:.2f}, pitch={pitch_applied_offset:.2f}, roll={roll_applied_offset:.2f})")
    print("")

def main():
    # command_yaw = 10.0
    # command_pitch = 5

    # offset_yaw = 10
    # offset_pitch = 0
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    

    # command_yaw = 50.0
    # command_pitch = -10

    # offset_yaw = 10
    # offset_pitch = 0
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    

    print("---offset calculations and estimations ---")
    yaw_commands = [0, 30, 60, 90, 120, 150, 180]
    pitch_commands = [10]
    offset_yaw = 0
    offset_pitch = -1
    offset_roll = 0

    for pitch_command in pitch_commands:
        for yaw_command in yaw_commands:
            calculate_commands_and_print(yaw_command,
                                 pitch_command,
                                 offset_pitch,
                                 offset_yaw,
                                 offset_roll)


    # command_yaw = 0.0
    # command_pitch = 10

    
    
    
    # command_yaw = 30.0
    # command_pitch = 10

    # offset_yaw = 0
    # offset_pitch = 10
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    
    # command_yaw = 60.0
    # command_pitch = 10

    # offset_yaw = 0
    # offset_pitch = 10
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)

    # command_yaw = 90.0
    # command_pitch = 10

    # offset_yaw = 0
    # offset_pitch = 10
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)

    # command_yaw = 120.0
    # command_pitch = 10

    # offset_yaw = 0
    # offset_pitch = 10
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    
    # command_yaw = 150.0
    # command_pitch = 10

    # offset_yaw = 0
    # offset_pitch = 10
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    
    # command_yaw = 180.0
    # command_pitch = 10

    # offset_yaw = 0
    # offset_pitch = 10
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    

    # command_yaw = 30.0
    # command_pitch = -30

    # offset_yaw = 30
    # offset_pitch = 0
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    
    # command_yaw = 30.0
    # command_pitch = -20

    # offset_yaw = 30
    # offset_pitch = 0
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    
    # command_yaw = 30.0
    # command_pitch = -10

    # offset_yaw = 30
    # offset_pitch = 0
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)

    # command_yaw = 30.0
    # command_pitch = 0

    # offset_yaw = 30
    # offset_pitch = 0
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)

    # command_yaw = 30.0
    # command_pitch = 10

    # offset_yaw = 30
    # offset_pitch = 10
    # offset_roll = 0
    # calculate_commands_and_print(command_yaw,
    #                              command_pitch,
    #                              offset_pitch,
    #                              offset_yaw,
    #                              offset_roll)
    

if __name__ == '__main__':
    main()