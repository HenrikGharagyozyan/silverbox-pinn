"""Loading and splitting of the Silverbox excerpt (SNLS80mV.mat).

Channel V1 is the input voltage u(t), channel V2 is the output voltage y(t).
The excerpt is the slice [42650:46746] (4096 samples), centered by the means of
u and y over that slice, then split into 3072 training and 1024 test samples.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SNLS80mV.mat"

FS = 610.35  # sampling frequency [Hz]
DT = 1.0 / FS  # sampling step [s]

START = 42650
STOP = 46746  # exclusive, STOP - START = 4096
N_TRAIN = 3072
N_TEST = 1024


@dataclass
class SilverboxData:
    t: np.ndarray  # time of each excerpt sample [s], starting at 0
    u: np.ndarray  # centered input on the whole excerpt
    y: np.ndarray  # centered output on the whole excerpt
    u_mean: float
    y_mean: float
    dt: float = DT
    n_train: int = N_TRAIN
    n_test: int = N_TEST

    @property
    def u_train(self) -> np.ndarray:
        return self.u[: self.n_train]

    @property
    def y_train(self) -> np.ndarray:
        return self.y[: self.n_train]

    @property
    def u_test(self) -> np.ndarray:
        return self.u[self.n_train :]

    @property
    def y_test(self) -> np.ndarray:
        return self.y[self.n_train :]


def load_silverbox(path: Path | str = DATA_PATH) -> SilverboxData:
    """Load the excerpt, subtract the excerpt means, and return train/test views."""
    mat = loadmat(path)
    u_raw = np.asarray(mat["V1"], dtype=np.float64).ravel()[START:STOP]
    y_raw = np.asarray(mat["V2"], dtype=np.float64).ravel()[START:STOP]
    assert u_raw.size == y_raw.size == N_TRAIN + N_TEST

    u_mean = float(u_raw.mean())
    y_mean = float(y_raw.mean())
    u = u_raw - u_mean
    y = y_raw - y_mean
    t = np.arange(u.size) * DT
    return SilverboxData(t=t, u=u, y=y, u_mean=u_mean, y_mean=y_mean)
