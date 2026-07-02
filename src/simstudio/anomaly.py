from collections import deque
import math

class ZScoreAnomaly:
    def __init__(self, window: int = 250, z_thresh: float = 4.0):
        self.window = max(30, window)
        self.z_thresh = z_thresh
        self.buf = deque(maxlen=self.window)

    def update(self, x: float):
        self.buf.append(x)
        if len(self.buf) < 40:
            return {"ready": False, "z": 0.0, "is_anomaly": False}
        mu = sum(self.buf)/len(self.buf)
        var = sum((v-mu)*(v-mu) for v in self.buf)/len(self.buf)
        sd = math.sqrt(max(1e-9, var))
        z = (x-mu)/sd
        return {"ready": True, "z": z, "is_anomaly": abs(z) >= self.z_thresh, "mu": mu, "sd": sd}
