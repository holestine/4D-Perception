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
        """Set up a constant-velocity Kalman filter seeded from the first detection.

        Parameters
        ----------
        box : ndarray (7,)
            Initial [x, y, z, l, w, h, yaw] in world/LiDAR frame.
        velocity_process_noise : float
            Multiplier on the velocity rows of Q.  Higher values let the filter
            adapt to acceleration faster at the cost of noisier velocity estimates.
        dt : float
            Seconds between frames — used to couple position and velocity in F.
        """
        self.kf = KalmanFilter(dim_x=10, dim_z=7)

        # Observe only the 7 pose dimensions; velocities are latent.
        self.kf.H = np.eye(10)[0:7]

        # Constant-velocity model: x_{t+1} = x_t + v_t * dt.
        self.kf.F = np.eye(10)
        self.kf.F[0:3, 7:10] = dt * np.eye(3)

        # Seed position/shape from the first detection; velocities start at 0.
        self.kf.x[:7] = box[:7].reshape(7, 1)

        # High velocity uncertainty at birth — we have no prior on how fast the
        # object is moving, so let the filter learn it quickly from the first few
        # updates rather than trusting the zero initialization.
        self.kf.P = np.eye(10)
        self.kf.P[7:10, 7:10] *= 100

        # Process noise Q: how much we expect the true state to jitter per frame.
        # Scaled by dt so the budget grows with step size — a vehicle is harder to
        # predict over 0.5 s (nuScenes 2 Hz) than over 0.1 s (KITTI 10 Hz).
        # Shape (l, w, h, yaw) tightened to 5% — a car's size doesn't change
        # frame-to-frame; keeping this small prevents the filter from "shrinking"
        # or "growing" boxes to absorb pose residuals.
        # Velocity rows scaled by velocity_process_noise (default 1.0) —
        # tunable so aggressive maneuvers can be tracked without over-smoothing.
        self.kf.Q = np.eye(10) * dt
        self.kf.Q[3:7, 3:7]   *= 0.05
        self.kf.Q[7:10, 7:10] *= velocity_process_noise

        # Measurement noise R: detector uncertainty in each observed dimension.
        # Position noisier (1.0) than shape (0.5) — localization varies with
        # range and occlusion, but shape estimates from a good detector are stable.
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
