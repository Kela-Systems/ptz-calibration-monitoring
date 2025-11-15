import numpy as np
from scipy.spatial.transform import Rotation

def main():
    R_physical_bad = np.array([
    [ 0.9398, -0.1572,  0.3034],
    [ 0.2187,  0.9589, -0.1807],
    [-0.2625,  0.2362,  0.9356]
    ])

    R_telemetry_bad = np.array([
        [ 0.9397, -0.1445, -0.3100],
        [ 0.2086,  0.9604,  0.1848],
        [ 0.2710, -0.2383,  0.9326]
    ])

    # Calculate the hypothetical R_flip
    R_flip = R_physical_bad.T @ R_telemetry_bad

    print("Calculated R_flip:\n", np.round(R_flip, 3))

    # Decompose it to understand what it is
    yaw, pitch, roll = Rotation.from_matrix(R_flip).as_euler('yxz', degrees=True)
    print(f"\nThis R_flip is a rotation of:")
    print(f"  Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")

if __name__ == '__main__':
    main()