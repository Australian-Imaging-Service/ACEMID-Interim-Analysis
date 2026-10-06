"""Parse Vectra WB360 .sfcm camera-model files.

Each .sfcm is plain text, e.g.:

    #SFCM 1.2
    pos 755.19 153.62 567.41
    rot (0.59,0.312,-0.395,0.630)
    f   -40.04
    pp  2639.48 1739.05
    K1  4.58e-05
    K2  -5.66e-08
    pixSize 0.0043 0.0043
    imDomain [0,5202)x[0,3465)
    units mm

Provides a Camera dataclass with intrinsics (pixels) and extrinsics, plus
helpers to resolve the (unknown) rotation convention empirically.
"""
from __future__ import annotations
import re
import glob
import os
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    name: str
    pos: np.ndarray        # (3,) camera center in world, mm  -- as written in file
    quat: np.ndarray       # (4,) raw quaternion as written (order TBD)
    f_mm: float            # focal length mm (may be negative in file)
    pp: np.ndarray         # (2,) principal point in pixels
    k1: float
    k2: float
    pix_mm: np.ndarray     # (2,) pixel size mm
    width: int
    height: int

    @property
    def f_px(self) -> np.ndarray:
        return np.array([abs(self.f_mm) / self.pix_mm[0],
                         abs(self.f_mm) / self.pix_mm[1]])


def _floats(s: str):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def parse_sfcm(path: str) -> Camera:
    name = os.path.splitext(os.path.basename(path))[0]
    pos = quat = pp = pix = None
    f_mm = k1 = k2 = 0.0
    w = h = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("pos"):
                pos = np.array(_floats(line)[:3])
            elif line.startswith("rot"):
                quat = np.array(_floats(line)[:4])
            elif line.startswith("f ") or line == "f" or re.match(r"^f\s", line):
                v = _floats(line)
                if v:
                    f_mm = v[0]
            elif line.startswith("pp"):
                pp = np.array(_floats(line)[:2])
            elif line.startswith("K1"):
                k1 = _floats(line)[0]
            elif line.startswith("K2"):
                k2 = _floats(line)[0]
            elif line.startswith("pixSize"):
                pix = np.array(_floats(line)[:2])
            elif line.startswith("imDomain"):
                nums = _floats(line)
                # [0,W)x[0,H)
                w = int(round(nums[1]))
                h = int(round(nums[3]))
    return Camera(name, pos, quat, f_mm, pp, k1, k2, pix, w, h)


def load_calib_dir(calib_dir: str) -> dict[str, Camera]:
    cams = {}
    for p in sorted(glob.glob(os.path.join(calib_dir, "*.sfcm"))):
        c = parse_sfcm(p)
        cams[c.name] = c
    return cams


# ---- quaternion helpers -------------------------------------------------

def quat_to_R(q, order):
    """Return 3x3 rotation matrix. order is e.g. 'wxyz' or 'xyzw'."""
    idx = {ch: i for i, ch in enumerate(order)}
    w = q[idx['w']]; x = q[idx['x']]; y = q[idx['y']]; z = q[idx['z']]
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    return R
