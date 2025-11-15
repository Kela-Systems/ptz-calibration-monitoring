import numpy as np

from .utils import calculate_angular_dist
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

def cost_function(r_align_rotvec, R_phys, R_tel):
    """
    Finds the angular distance between the measurement and the prediction
    for a given candidate R_align.
    """
    R_align = Rotation.from_rotvec(r_align_rotvec).as_matrix()
    
    # Predict what the sensor should have seen
    R_sensor_predicted = R_align.T @ R_tel @ R_align
    
    # Calculate the error between prediction and the actual measurement
    return calculate_angular_dist(R_phys, R_sensor_predicted)


def main():
    R_physical = np.array([
    [ 0.93981318, -0.15718712,  0.30338656],
    [ 0.21871873,  0.95890763, -0.18071603],
    [-0.26251346,  0.23619563,  0.93557379]
    ])

    R_telemetry = np.array([
    [ 0.93969274, -0.14448939, -0.3100006 ],
    [ 0.20863415,  0.96037542,  0.18479945],
    [ 0.27101539, -0.23833141,  0.93260324]
    ])

    original_angular_distance = calculate_angular_dist(R_physical, R_telemetry)
    print("Original angular distance: ", original_angular_distance)

    rot_physical = Rotation.from_matrix(R_physical)
    rot_telemetry = Rotation.from_matrix(R_telemetry)

    axis_physical = rot_physical.as_rotvec() / rot_physical.magnitude()
    axis_telemetry = rot_telemetry.as_rotvec() / rot_telemetry.magnitude()

    print("--- PROOF: The Axes of Rotation are Different ---")
    print("Axis of R_physical: ", np.round(axis_physical, 4))
    print("Axis of R_telemetry:", np.round(axis_telemetry, 4))

    cos_angle = np.dot(axis_physical, axis_telemetry)
    angle_between_axes = np.degrees(np.arccos(cos_angle))

    print(f"\nAngle between the two axes of rotation: {angle_between_axes:.2f} degrees")

    print("\n--- PROOF: Solving for the Optimal R_align for this specific pair ---")
    print("Searching for the R_align that best explains the 41.84 degree error...")

    initial_guess = np.array([0.0, 0.0, 0.0])

    # Run the optimization
    result = minimize(
        cost_function,
        initial_guess,
        args=(R_physical, R_telemetry),
        method='L-BFGS-B'
    )

    if result.success:
        # Get the best R_align found by the optimizer
        R_align_optimal = Rotation.from_rotvec(result.x).as_matrix()
        
        # The final, minimized error
        minimized_error = result.fun

        print("\nOptimization successful!")
        print("Found an optimal R_align matrix:\n", np.round(R_align_optimal, 4))
        
        # Decompose the optimal R_align to understand it
        yaw, pitch, roll = Rotation.from_matrix(R_align_optimal).as_euler('yxz', degrees=True)
        print(f"\nThis R_align represents a physical misalignment of:")
        print(f"  Yaw:   {yaw:.2f} degrees")
        print(f"  Pitch: {pitch:.2f} degrees")
        print(f"  Roll:  {roll:.2f} degrees")

        print(f"\nOriginal angular distance between R_physical and R_telemetry: {original_angular_distance:.2f} degrees")
        print(f"NEW minimized angular distance after applying this optimal R_align: {minimized_error:.4f} degrees")
    else:
        print("Optimization failed:", result.message)




if __name__ == '__main__':
    main()