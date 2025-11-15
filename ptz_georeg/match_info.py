from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional

@dataclass
class MatchInfo:
    """
    A class to store all relevant information about a feature matching result
    between a query and a reference frame.
    """
    # --- Core Match Data ---
    ref_frame: Optional[Dict[str, Any]] = None
    num_inliers: int = -1
    homography_matrix: Optional[np.ndarray] = None

    # --- Inlier Details (for debugging or visualization) ---
    mkpts_query: Optional[np.ndarray] = None  # Matched keypoints from query (inliers)
    mkpts_ref: Optional[np.ndarray] = None  # Matched keypoints from reference (inliers)
    
    # --- Raw Match Statistics (before RANSAC) ---
    num_raw_matches: int = -1
    cumulative_score: float = -1.0
    
    def __post_init__(self):
        """Ensure numpy arrays are handled correctly for equality checks."""
        if self.homography_matrix is not None:
            self.homography_matrix = np.array(self.homography_matrix)
        if self.mkpts_query is not None:
            self.mkpts_query = np.array(self.mkpts_query)
        if self.mkpts_ref is not None:
            self.mkpts_ref = np.array(self.mkpts_ref)

    @property
    def is_valid(self) -> bool:
        """A simple property to check if the match is considered valid."""
        return self.ref_frame is not None and self.num_inliers > 0
    
@dataclass
class SensorTelemetryPair:
    """
    A class to store pairs of sensor and telemetry rotation matrices.
    """
    # --- Core Match Data ---
    sensor_rotation_matrix:np.ndarray = None
    telemetry_rotation_matrix:np.ndarray = None
    sensor_translation_matrix:np.ndarray = None
    telemetry_translation_matrix:np.ndarray = None

@dataclass
class FrameMatchingPointsStatistics:
    def __init__(self):
        self.filename:str = ''
        self.avg_number_inliers:float = 0.0 
        self.max_number_inliers:int = 0
    