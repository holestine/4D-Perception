import numpy as np
from filterpy.kalman import KalmanFilter


class Obstacle3D:
    """Individual 3D track using a constant-velocity Kalman filter.

    State  (10-D): [x, y, z, l, w, h, yaw, vx, vy, vz]
    Measurement (7-D): [x, y, z, l, w, h, yaw]
    """

    def __init__(self, box, score, track_id, velocity_process_noise=1.0, dt=0.1, name=None):
        """
        Parameters
        ----------
        box      : ndarray (7,)  [x, y, z, l, w, h, yaw] in world/LiDAR frame
        score    : float  detector confidence
        track_id : int    unique ID assigned by the tracker
        velocity_process_noise : float
            Scale on the velocity rows of Q — higher = faster adaptation.
        dt   : float  seconds between frames (default 0.1 = 10 Hz LiDAR)
        name : str    detection class label, used for class-gated association
        """
        self.id                = track_id
        self.name              = name
        self.time_since_update = 0
        self.hit_streak        = 0
        self.score             = score
        self._init_kalman(box, velocity_process_noise, dt)

    def _init_kalman(self, box, velocity_process_noise, dt):
        self.kf = KalmanFilter(dim_x=10, dim_z=7)

        self.kf.H = np.eye(10)[0:7]

        self.kf.F = np.eye(10)
        self.kf.F[0:3, 7:10] = dt * np.eye(3)

        self.kf.x[:7] = box[:7].reshape(7, 1)

        self.kf.P = np.eye(10)
        self.kf.P[7:10, 7:10] *= 100

        self.kf.Q = np.eye(10) * 0.1
        self.kf.Q[3:7, 3:7]   *= 0.05
        self.kf.Q[7:10, 7:10] *= velocity_process_noise

        self.kf.R = np.eye(7) * 1.0
        self.kf.R[3:7, 3:7] *= 0.5

    def predict(self):
        """Advance by one step; return predicted 7-D box."""
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x[:7].reshape(-1)

    def update(self, box, score):
        """Correct with a matched detection."""
        self.time_since_update = 0
        self.hit_streak       += 1
        self.score             = score
        self.kf.update(box[:7].reshape(7, 1))

    def get_state(self):
        """Return current filtered 7-D box estimate."""
        return self.kf.x[:7].reshape(-1)
