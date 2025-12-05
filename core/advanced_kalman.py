"""
Advanced Kalman filtering for robust facial landmark tracking.

This module implements high-quality tracking techniques used in professional VTuber software:
- Per-landmark confidence weighting based on velocity
- Outlier detection and rejection
- Region-based smoothing (stable anchors vs expressive features)
- Velocity clamping to prevent jitter
- Confidence-weighted Kalman gain
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# MediaPipe landmark indices for different facial regions
class LandmarkRegions:
    """Defines landmark regions with different stability characteristics."""
    
    # STABLE ANCHORS - Rock-solid anatomical points
    STABLE = [
        # Inner eye corners
        133, 362,
        # Nose bridge
        6, 197, 195, 168,
        # Nose tip
        1, 4,
        # Brow anchors
        70, 300, 107, 336,
        # Outer eye corners (relatively stable)
        33, 263,
    ]
    
    # MEDIUM STABILITY - Cheek and jaw root
    MEDIUM = [
        # Cheek landmarks
        234, 454, 123, 352,
        # Upper jaw
        172, 397, 58, 288,
    ]
    
    # EXPRESSIVE - Highly flexible, responsive features
    EXPRESSIVE = [
        # Lips (all lip landmarks)
        61, 291, 0, 17, 84, 314, 405, 181,
        78, 308, 95, 324, 13, 14, 87, 317,
        # Eyelids
        159, 145, 386, 374, 160, 144, 387, 373,
        # Jaw outline
        152, 377, 400, 378, 379, 365, 397, 288,
        # Eyebrows (expressive)
        66, 105, 63, 296, 334, 293,
    ]
    
    @staticmethod
    def get_region_weight(landmark_idx: int, 
                         stable_weight: float = 0.8,
                         medium_weight: float = 0.5,
                         expressive_weight: float = 0.2) -> float:
        """Get smoothing weight for a landmark based on its region."""
        if landmark_idx in LandmarkRegions.STABLE:
            return stable_weight
        elif landmark_idx in LandmarkRegions.MEDIUM:
            return medium_weight
        elif landmark_idx in LandmarkRegions.EXPRESSIVE:
            return expressive_weight
        else:
            # Default: medium weight for unlabeled landmarks
            return medium_weight


class AdvancedKalmanFilter:
    """
    Enhanced Kalman filter with confidence weighting and outlier rejection.
    
    This is the core of robust tracking stability.
    """
    
    def __init__(self, x0: float = 0.0, v0: float = 0.0, dt: float = 1.0/30.0,
                 q: float = 1e-2, r: float = 1e-2,
                 min_confidence: float = 0.1,
                 velocity_scale: float = 0.05,
                 outlier_threshold: float = 0.05,
                 max_velocity: float = 0.1):
        """
        Initialize advanced Kalman filter.
        
        Args:
            x0: Initial position
            v0: Initial velocity
            dt: Time step (1/FPS)
            q: Process noise covariance
            r: Measurement noise covariance
            min_confidence: Minimum confidence value
            velocity_scale: Scale factor for velocity → confidence conversion
            outlier_threshold: Distance threshold for outlier detection
            max_velocity: Maximum allowed velocity (for clamping)
        """
        self.dt = dt
        self.x = np.array([[x0], [v0]], dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32) * 1.0
        self.F = np.array([[1, dt], [0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0]], dtype=np.float32)
        self.Q = np.array([[q, 0], [0, q]], dtype=np.float32)
        self.R = np.array([[r]], dtype=np.float32)
        
        # Advanced tracking parameters
        self.min_confidence = min_confidence
        self.velocity_scale = velocity_scale
        self.outlier_threshold = outlier_threshold
        self.max_velocity = max_velocity
        
        # Previous measurement for velocity calculation
        self.prev_measurement = x0
        self.measurement_velocity = 0.0
        
    def predict(self) -> float:
        """Predict next state and return predicted position."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0])
    
    def get_velocity(self) -> float:
        """Get current velocity estimate from filter state."""
        return float(self.x[1, 0])
    
    def compute_confidence(self, measurement: float) -> float:
        """
        Compute confidence score based on measurement velocity.
        
        Fast-moving points get low confidence (likely noise).
        Slow-moving points get high confidence (stable).
        
        Returns:
            Confidence value between min_confidence and 1.0
        """
        # Calculate velocity from measurement change
        self.measurement_velocity = abs(measurement - self.prev_measurement)
        self.prev_measurement = measurement
        
        # Confidence = 1 / (1 + velocity)
        # Scaled by velocity_scale to tune sensitivity
        confidence = 1.0 / (1.0 + self.measurement_velocity / self.velocity_scale)
        
        # Clamp to minimum confidence
        return max(self.min_confidence, confidence)
    
    def is_outlier(self, measurement: float, predicted: float) -> bool:
        """
        Detect if measurement is an outlier.
        
        Args:
            measurement: Current measurement
            predicted: Predicted value from Kalman filter
            
        Returns:
            True if measurement is likely an outlier
        """
        distance = abs(measurement - predicted)
        return distance > self.outlier_threshold
    
    def clamp_velocity(self, measurement: float, predicted: float) -> float:
        """
        Clamp velocity to prevent excessive jumps.
        
        Args:
            measurement: Raw measurement
            predicted: Predicted value
            
        Returns:
            Clamped measurement
        """
        delta = measurement - predicted
        
        if abs(delta) > self.max_velocity:
            # Clamp the movement
            clamped_delta = np.sign(delta) * self.max_velocity
            return predicted + clamped_delta
        
        return measurement
    
    def update(self, z: float, confidence: float = 1.0, 
               reject_outliers: bool = True,
               clamp_velocity: bool = True) -> Tuple[float, float]:
        """
        Update filter with measurement and confidence weighting.
        
        Args:
            z: Measurement value
            confidence: Confidence weight (0.0 to 1.0)
            reject_outliers: Whether to reject outlier measurements
            clamp_velocity: Whether to clamp excessive velocities
            
        Returns:
            Tuple of (filtered_value, confidence_used)
        """
        # Get prediction first
        predicted = float(self.x[0, 0])
        
        # Check for outliers
        if reject_outliers and self.is_outlier(z, predicted):
            # Reject outlier - trust prediction instead
            return predicted, 0.0
        
        # Clamp velocity if enabled
        if clamp_velocity:
            z = self.clamp_velocity(z, predicted)
        
        # Standard Kalman update with confidence weighting
        z_array = np.array([[z]], dtype=np.float32)
        y = z_array - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Apply confidence weighting to Kalman gain
        # Lower confidence = trust measurement less, trust prediction more
        K_weighted = confidence * K
        
        self.x = self.x + K_weighted @ y
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K_weighted @ self.H) @ self.P
        
        return float(self.x[0, 0]), confidence
    
    def step(self, z: float, 
             region_weight: float = 0.5,
             use_confidence: bool = True,
             reject_outliers: bool = True,
             clamp_velocity: bool = True) -> float:
        """
        Complete filter step: predict + update with all features.
        
        Args:
            z: Measurement value
            region_weight: Smoothing weight for this landmark's region
            use_confidence: Whether to use velocity-based confidence
            reject_outliers: Whether to reject outliers
            clamp_velocity: Whether to clamp velocities
            
        Returns:
            Filtered position value
        """
        # Predict
        predicted = self.predict()
        
        # Compute confidence if enabled
        if use_confidence:
            confidence = self.compute_confidence(z)
            # Blend confidence with region weight
            # Region weight acts as a base smoothing level
            final_confidence = confidence * (1.0 - region_weight) + region_weight
        else:
            final_confidence = 1.0
        
        # Update with confidence
        filtered, _ = self.update(z, final_confidence, reject_outliers, clamp_velocity)
        
        # Blend Kalman output with raw measurement based on confidence
        # High confidence → trust Kalman more
        # Low confidence → trust raw measurement more
        if use_confidence:
            blended = final_confidence * filtered + (1.0 - final_confidence) * z
            return blended
        else:
            return filtered


def compute_landmark_velocities(current_pts: List[Tuple[float, float]], 
                                previous_pts: List[Tuple[float, float]]) -> List[float]:
    """
    Compute velocities for all landmarks.
    
    Args:
        current_pts: Current frame landmarks [(x, y), ...]
        previous_pts: Previous frame landmarks [(x, y), ...]
        
    Returns:
        List of velocity magnitudes for each landmark
    """
    velocities = []
    for curr, prev in zip(current_pts, previous_pts):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        velocity = np.sqrt(dx*dx + dy*dy)
        velocities.append(velocity)
    return velocities
