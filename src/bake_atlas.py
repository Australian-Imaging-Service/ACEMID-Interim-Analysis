#!/usr/bin/env python3
"""Bake Vectra's own texture atlas from a .tom into a multi-texture GLB.

Decoded .tom texture model:
  Vertices : float32 XYZ (world mm)
  Tris____ : zlib; faces = idx[0::2].reshape(-1,3)
  TexVertA : 191405 texture-vertices, each (f32 u, f32 v, i32 page)  u,v in [0,1]
  CnrTexVs : zlib; per-corner triple (i32 face_index, i32 corner#0-2, i32 texvert)
  TxtrJPG_ + 46 TxtrJPGA : 47 JPEG atlas pages (~5202x2836)

Usage: bake_atlas.py <tom> <out_dir> [--maxw 3500]
"""
import sys, os, zlib, struct, argparse
import numpy as np
import trimesh
from PIL import Image
import io


def find_chunks(data):
    """Return dict tag->list of (data_off,len) by walking the container."""
    import re
    N = len(data)
    out = {}
    pos = 8
    def tag_ok(b):
        nm = b[:8].rstrip(b'_').rstrip(b'.')
        if not nm or not all(32 < c < 127 for c in b[:8]):
            return None
        r = b[8:16].rstrip(b'.')
        if len(r) > 3 or not all(32 < c < 127 for c in r):
            return None
        return b[:8].rstrip(b'.').rstrip(b'_').decode('latin1')
    guard = 0
    pages = []
    while pos + 24 <= N and guard < 300:
        guard += 1
        length = struct.unpack_from('<Q', data, pos)[0]
        t = tag_ok(data[pos + 8:pos + 24])
        if t is None or length < 0 or pos + 24 + length > N:
            nxt = None
            for p in range(pos + 1, min(pos + 64, N - 24)):
                l2 = struct.unpack_from('<Q', data, p)[0]
                if tag_ok(data[p + 8:p + 24]) and 0 <= l2 and p + 24 + l2 <= N:
                    nxt = p; break
            if nxt is None:
                break
            pos = nxt; continue
        d0 = pos + 24
        if t in ('TxtrJPG', 'TxtrJPGA'):
            pages.append((d0, length))
        out.setdefault(t, []).append((d0, length))
        pos = d0 + length
        if t.startswith('EndTOM'):
            break
    out['_pages'] = pages
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tom'); ap.add_argument('out_dir')
    ap.add_argument('--maxw', type=int, default=3500)
    ap.add_argument('--quality', type=int, default=88)
    ap.add_argument('--name', default='vectra_atlas.glb')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = open(a.tom, 'rb').read()
    ch = find_chunks(data)

    vo, vl = ch['Vertices'][0]
    nV = vl // 12
    V = np.frombuffer(data, '<f4', nV * 3, vo).reshape(-1, 3).astype(np.float32)
    z = data.find(b'\x78\x9c', vo + vl)
    faces = np.frombuffer(zlib.decompressobj().decompress(data[z:]), '<i4')[0::2].reshape(-1, 3)
    nF = len(faces)
    print(f"verts {nV}  faces {nF}")

    to, tl = ch['TexVertA'][0]
    TV = np.frombuffer(zlib.decompress(data[to:to + tl]),
                       dtype=[('u', '<f4'), ('v', '<f4'), ('p', '<i4')])
    co, cl = ch['CnrTexVs'][0]
    trip = np.frombuffer(zlib.decompress(data[co:co + cl]), '<i4')
    trip = trip[:(len(trip) // 3) * 3].reshape(-1, 3)
    fidx, cno, tvi = trip[:, 0], trip[:, 1], trip[:, 2]

    # per-(geometry-face) uv[3,2] and page
    uvf = np.zeros((nF, 3, 2), np.float32)
    pagef = np.full(nF, -1, np.int32)
    uvf[fidx, cno, 0] = TV['u'][tvi]
    uvf[fidx, cno, 1] = TV['v'][tvi]
    pagef[fidx] = TV['p'][tvi]
    has = pagef >= 0
    print(f"textured faces {has.sum()} / {nF}")

    pages = ch['_pages']
    print(f"{len(pages)} texture pages")

    scene = trimesh.Scene()
    for pg in range(len(pages)):
        sel = np.where(has & (pagef == pg))[0]
        if len(sel) == 0:
            continue
        d0, ln = pages[pg]
        img = Image.open(io.BytesIO(data[d0:d0 + ln])).convert('RGB')
        W, H = img.size
        uv = uvf[sel].reshape(-1, 2).copy()     # normalized, v up
        # crop page to used-UV bbox (+pad) to shrink the texture massively
        pad = 0.01
        umin, umax = max(0, uv[:, 0].min() - pad), min(1, uv[:, 0].max() + pad)
        vmin, vmax = max(0, uv[:, 1].min() - pad), min(1, uv[:, 1].max() + pad)
        x0, x1 = int(umin * W), int(np.ceil(umax * W))
        y0, y1 = int((1 - vmax) * H), int(np.ceil((1 - vmin) * H))   # v-flip for pixels
        crop = img.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        if cw > a.maxw:
            nh = max(1, int(ch * a.maxw / cw))
            crop = crop.resize((a.maxw, nh), Image.LANCZOS)
        # re-encode as JPEG so trimesh embeds JPEG (not lossless PNG) in the GLB
        buf = io.BytesIO(); crop.save(buf, 'JPEG', quality=a.quality); buf.seek(0)
        crop = Image.open(buf)
        # remap uv into the crop. Keep trimesh (v-up) convention -- trimesh flips
        # to glTF v-down on export, so DO NOT pre-flip here (double-flip = bug).
        nu = (uv[:, 0] - umin) / max(1e-9, (umax - umin))
        nv = (uv[:, 1] - vmin) / max(1e-9, (vmax - vmin))
        uv2 = np.stack([nu, nv], 1)
        verts = V[faces[sel].reshape(-1)]
        f = np.arange(len(sel) * 3).reshape(-1, 3)
        vis = trimesh.visual.TextureVisuals(uv=uv2, image=crop)
        # white base-colour factor (no grey tint) + double-sided (no black backfaces)
        vis.material.baseColorFactor = np.array([255, 255, 255, 255], np.uint8)
        vis.material.doubleSided = True
        m = trimesh.Trimesh(vertices=verts, faces=f, visual=vis, process=False)
        m.vertex_normals          # force normal computation so they export
        scene.add_geometry(m, geom_name=f'page{pg}')
    glb = os.path.join(a.out_dir, a.name)
    scene.export(glb)
    print("wrote", glb, f"({os.path.getsize(glb)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
