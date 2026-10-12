"""Parse a Vectra .tom container into its chunks.

Layout (reverse-engineered, verified against full session files):

  8-byte magic:  E8 'T' 'O' 'M' 0D 0A 20 0A
      The \\r\\n \\n guard is the same trick PNG uses to detect FTP text-mode
      mangling, which is the first clue this is a properly designed container.

  then repeated chunks:
      uint64  data_length          (little-endian)
      8-byte  name                 ASCII, '.'/'_' padded  e.g. 'Vertices'
      8-byte  qualifier            ASCII, '.' padded      e.g. 'z', 'tz', ''
      bytes   data[data_length]
      uint32  trailer              always 0 in every file seen (reserved)

  The 4-byte trailer is the part that is easy to get wrong: an earlier version of
  this parser assumed 8-byte alignment instead, which happens to survive the
  first two chunks of a session file and then desynchronises. The check that the
  layout is right is that walking it consumes the file EXACTLY -- last chunk is
  'EndTOM' and 0 bytes remain.

  The qualifier carries the compression flag: a qualifier containing 'z' means
  the payload is a raw zlib stream (Tris____ is 'tz', TexVertA / CnrTexVs are
  'z'). Sniffing for a 0x78 first byte also works but the flag is authoritative.

Typical session inventory (58 chunks):
  Header, ThumbPNG (a PNG preview), Metadata, Vertices (f32 XYZ, world mm),
  Tris (zlib), TxtrJPG + 46 x TxtrJPGA (JPEG atlas pages), TexAtlas, AtlasIms,
  TexVertA (zlib; u,v,page), CnrTexVs (zlib; face,corner,texvert), Trnsform,
  EndTOM.
"""
import sys, struct, os, zlib

MAGIC = bytes([0xE8]) + b'TOM\r\n \n'
TRAILER = 4          # bytes of reserved padding after each chunk's data


def parse(path):
    """Yield (name, qualifier, data_offset, length, blob) for every chunk."""
    data = open(path, 'rb').read()
    assert data[:8] == MAGIC, f"bad magic {data[:8]!r}"
    off = 8
    chunks = []
    while off + 24 <= len(data):
        (length,) = struct.unpack_from('<Q', data, off)
        raw = data[off + 8:off + 24]
        name = raw[:8].rstrip(b'._').decode('latin1', 'replace')
        qual = raw[8:].rstrip(b'.').decode('latin1', 'replace')
        start = off + 24
        end = start + length
        if end + TRAILER > len(data):
            print(f"  !! chunk {name} length {length} exceeds file", file=sys.stderr)
            break
        chunks.append((name, qual, start, length, data[start:end]))
        off = end + TRAILER
        if name.startswith('EndTOM'):
            break
    leftover = len(data) - off
    if leftover:
        print(f"  !! {leftover} bytes left unconsumed - layout assumption is off",
              file=sys.stderr)
    return chunks


def inflate(qual, blob):
    """Decompress a chunk payload if it is flagged (or looks) zlib-compressed."""
    if 'z' not in qual and blob[:1] != b'\x78':
        return None
    try:
        return zlib.decompress(blob)
    except Exception:
        # some payloads are a zlib stream followed by trailing bytes
        try:
            return zlib.decompressobj().decompress(blob)
        except Exception as e:
            print(f"             zlib failed: {e}")
            return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <file.tom> [outdir]")
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else 'tom_chunks'
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for name, qual, start, length, blob in parse(path):
        n += 1
        print(f"{name:10s} {qual:4s} off={start:10d} len={length:10d}  "
              f"head={blob[:8].hex()}")
        safe = f"{n:02d}_{name}".replace('/', '_')
        with open(os.path.join(outdir, safe + '.bin'), 'wb') as f:
            f.write(blob)
        dec = inflate(qual, blob)
        if dec is not None:
            with open(os.path.join(outdir, safe + '.inflated'), 'wb') as f:
                f.write(dec)
            print(f"             -> zlib inflated to {len(dec)} bytes, "
                  f"head={dec[:16].hex()}")
    print(f"--- {n} chunks -> {outdir}")
