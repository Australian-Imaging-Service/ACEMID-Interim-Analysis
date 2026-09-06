#!/usr/bin/env python3
"""Trustworthy crisp render of the Vectra atlas-textured mesh.

Samples each triangle's colour from its assigned atlas page (full-res) and
rasterises with a painter's algorithm (cv2.fillConvexPoly). Bypasses any glTF
material pipeline so what you see is exactly the decoded atlas data.
"""
import sys, os, zlib, io
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_atlas import find_chunks

TOM = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
data = open(TOM, 'rb').read(); ch = find_chunks(data)
to, tl = ch['TexVertA'][0]
_tv = zlib.decompress(data[to:to + tl])
TV = np.frombuffer(_tv, dtype=[('u', '<f4'), ('v', '<f4'), ('p', '<i4')], count=len(_tv) // 12)
vo, vl = ch['Vertices'][0]; nV = vl // 12
V = np.frombuffer(data, '<f4', nV * 3, vo).reshape(-1, 3).astype(np.float64)
z = data.find(b'\x78\x9c', vo + vl)
faces = np.frombuffer(zlib.decompressobj().decompress(data[z:]), '<i4')[0::2].reshape(-1, 3)
nF = len(faces)
co, cl = ch['CnrTexVs'][0]
trip = np.frombuffer(zlib.decompress(data[co:co + cl]), '<i4')
trip = trip[:(len(trip) // 3) * 3].reshape(-1, 3)
uvf = np.zeros((nF, 3, 2)); pagef = np.full(nF, -1)
uvf[trip[:, 0], trip[:, 1], 0] = TV['u'][trip[:, 2]]
uvf[trip[:, 0], trip[:, 1], 1] = TV['v'][trip[:, 2]]
pagef[trip[:, 0]] = TV['p'][trip[:, 2]]
pages = ch['_pages']

# per-triangle colour: sample page image at centroid UV (full res)
tcol = np.full((nF, 3), 128, np.uint8)
for pg in range(len(pages)):
    sel = np.where(pagef == pg)[0]
    if not len(sel):
        continue
    d0, ln = pages[pg]
    im = np.asarray(Image.open(io.BytesIO(data[d0:d0 + ln])).convert('RGB'))
    H, W = im.shape[:2]
    uc = uvf[sel].mean(1)
    x = np.clip((uc[:, 0] * W).astype(int), 0, W - 1)
    y = np.clip(((1 - uc[:, 1]) * H).astype(int), 0, H - 1)
    tcol[sel] = im[y, x]
print("sampled triangle colours")


def render(size, ang, elev=0.0):
    c = V.mean(0)
    Rr = (V.max(0) - V.min(0)).max() * 1.35
    eye = c + np.array([Rr * np.cos(elev) * np.sin(ang), Rr * np.sin(elev),
                        Rr * np.cos(elev) * np.cos(ang)])
    up = np.array([0, 1.0, 0])
    f = c - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    Rmat = np.stack([s, u, f], 0)
    Xc = (V - eye) @ Rmat.T
    fpx = size * 1.15
    zc = Xc[:, 2]
    px = fpx * Xc[:, 0] / zc + size / 2
    py = -fpx * Xc[:, 1] / zc + size / 2
    P = np.stack([px, py], 1)
    img = np.full((size, size, 3), 255, np.uint8)
    tz = zc[faces].mean(1)
    vis = (zc[faces] > 0).all(1)
    order = np.argsort(-tz)
    order = order[vis[order]]
    pts = P[faces].astype(np.int32)          # (nF,3,2)
    col = tcol[:, ::-1]                        # RGB->BGR
    for fi in order:
        cv2.fillConvexPoly(img, pts[fi], (int(col[fi, 0]), int(col[fi, 1]), int(col[fi, 2])), cv2.LINE_AA)
    return img


for k in range(6):
    img = render(1000, 2 * np.pi * k / 6)
    cv2.imwrite(os.path.join(OUT, f'flat_view{k}.png'), img)
    print("rendered", k)
