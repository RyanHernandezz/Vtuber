import numpy as np

# --------- Small 1D Kalman filter helper (position, velocity) ----------
class KF1D:
    def __init__(self, x0=0.0, v0=0.0, dt=1.0/30.0, q=1e-2, r=1e-2):
        self.dt = dt
        self.x = np.array([[x0], [v0]], dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32) * 1.0
        self.F = np.array([[1, dt],[0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0]], dtype=np.float32)  # measure position only
        self.Q = np.array([[q, 0],[0, q]], dtype=np.float32)
        self.R = np.array([[r]], dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array([[z]], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        return float(self.x[0,0])

