#!/usr/bin/env python3
"""
mini_h5.py — minimal read-only HDF5 reader (superblock v0, v1 object headers,
symbol-table groups, contiguous/chunked datasets, optional deflate).

Written because h5py is unavailable in the verification sandbox. Scope is
deliberately narrow: enough to read the Microflown export. Every array it
returns is cross-checked in verify_h5_reader.py against an independent export.
"""
import struct, zlib, numpy as np


class H5:
    def __init__(self, path):
        self.f = open(path, 'rb')
        b = self._rd(0, 96)
        assert b[:8] == b'\x89HDF\r\n\x1a\n' and b[8] == 0, "only superblock v0 supported"
        self.O, self.L = b[13], b[14]
        root_ohdr = self._u(b, 56 + self.O, self.O)
        self.root = root_ohdr

    # ---- low-level ----
    def _rd(self, off, n):
        self.f.seek(off); return self.f.read(n)

    def _u(self, b, off, n):
        return int.from_bytes(b[off:off + n], 'little')

    # ---- object header v1 -> list of (type, bytes) ----
    def messages(self, addr):
        h = self._rd(addr, 16)
        assert h[0] == 1, f"object header v{h[0]} unsupported at {addr}"
        nmsg = struct.unpack('<H', h[2:4])[0]
        size = struct.unpack('<I', h[8:12])[0]
        blocks = [(addr + 16, size)]
        out = []
        while blocks and len(out) < nmsg:
            a, s = blocks.pop(0)
            blk = self._rd(a, s); p = 0
            while p + 8 <= s and len(out) < nmsg:
                mt, ms = struct.unpack('<HH', blk[p:p + 4])
                data = blk[p + 8:p + 8 + ms]
                if mt == 0x10:  # continuation
                    blocks.append((self._u(data, 0, self.O), self._u(data, self.O, self.L)))
                out.append((mt, data))
                p += 8 + ms
        return out

    # ---- groups ----
    def _heap_name(self, heap_addr, off):
        h = self._rd(heap_addr, 8 + 2 * self.L + self.O)
        assert h[:4] == b'HEAP'
        data_addr = self._u(h, 8 + 2 * self.L, self.O)
        s = self._rd(data_addr + off, 256)
        return s[:s.index(b'\x00')].decode()

    def _btree_group(self, addr, heap):
        n = self._rd(addr, 8 + 2 * self.O)
        assert n[:4] == b'TREE'
        level = n[5]; used = struct.unpack('<H', n[6:8])[0]
        p = 8 + 2 * self.O
        body = self._rd(addr + p, (used + 1) * self.L + used * self.O)
        kids = []
        q = self.L
        for _ in range(used):
            kids.append(self._u(body, q, self.O)); q += self.O + self.L
        res = {}
        for c in kids:
            if level > 0:
                res.update(self._btree_group(c, heap))
            else:
                sn = self._rd(c, 8)
                assert sn[:4] == b'SNOD'
                nsym = struct.unpack('<H', sn[6:8])[0]
                ent = 2 * self.O + 8 + 16
                raw = self._rd(c + 8, nsym * ent)
                for i in range(nsym):
                    e = raw[i * ent:(i + 1) * ent]
                    res[self._heap_name(heap, self._u(e, 0, self.O))] = self._u(e, self.O, self.O)
        return res

    def children(self, ohdr):
        for mt, d in self.messages(ohdr):
            if mt == 0x11:
                return self._btree_group(self._u(d, 0, self.O), self._u(d, self.O, self.O))
        raise KeyError("not a group")

    def get(self, path):
        addr = self.root
        for part in [p for p in path.split('/') if p]:
            addr = self.children(addr)[part]
        return addr

    # ---- datasets ----
    def dataset(self, path):
        addr = self.get(path)
        shape = dtype = layout = None; filters = []
        for mt, d in self.messages(addr):
            if mt == 0x01:                                   # dataspace
                ver, rank, flags = d[0], d[1], d[2]
                off = 8 if ver == 1 else 4
                shape = tuple(self._u(d, off + i * self.L, self.L) for i in range(rank))
            elif mt == 0x03:                                 # datatype
                cls = d[0] & 0x0F; size = struct.unpack('<I', d[4:8])[0]
                big = d[1] & 1
                if cls == 1: kind = 'f'
                elif cls == 0: kind = 'i' if (d[1] >> 3) & 1 else 'u'
                else: raise NotImplementedError(f"datatype class {cls}")
                dtype = np.dtype(('>' if big else '<') + kind + str(size))
            elif mt == 0x08:                                 # layout
                layout = d
            elif mt == 0x0B:                                 # filter pipeline
                ver, nf = d[0], d[1]
                p = 8 if ver == 1 else 2
                for _ in range(nf):
                    fid = struct.unpack('<H', d[p:p + 2])[0]
                    if ver == 1:
                        nl = struct.unpack('<H', d[p + 2:p + 4])[0]; ncv = struct.unpack('<H', d[p + 6:p + 8])[0]
                        p += 8 + ((nl + 7) // 8) * 8 + 4 * ncv + (4 if ncv % 2 else 0)
                    else:
                        ncv = struct.unpack('<H', d[p + 4:p + 6])[0]; p += 6 + 4 * ncv
                    filters.append(fid)
        return self._read(shape, dtype, layout, filters)

    def _read(self, shape, dtype, lay, filters):
        ver = lay[0]
        n = int(np.prod(shape)) if shape else 1
        if ver == 3:
            cls = lay[1]
            if cls == 1:                                     # contiguous
                a = self._u(lay, 2, self.O)
                return np.frombuffer(self._rd(a, n * dtype.itemsize), dtype).reshape(shape)
            if cls == 0:                                     # compact
                sz = struct.unpack('<H', lay[2:4])[0]
                return np.frombuffer(lay[4:4 + sz], dtype).reshape(shape)
            if cls == 2:                                     # chunked
                rank = lay[2] - 1
                bt = self._u(lay, 3, self.O)
                cdims = [struct.unpack('<I', lay[3 + self.O + 4 * i:7 + self.O + 4 * i])[0] for i in range(rank)]
                return self._read_chunked(bt, shape, dtype, cdims, filters)
        elif ver in (1, 2):
            rank = lay[1]; cls = lay[2]
            p = 8
            if cls == 1:
                a = self._u(lay, p, self.O)
                return np.frombuffer(self._rd(a, n * dtype.itemsize), dtype).reshape(shape)
            if cls == 2:
                bt = self._u(lay, p, self.O); p += self.O
                cdims = [struct.unpack('<I', lay[p + 4 * i:p + 4 * i + 4])[0] for i in range(rank - 1)]
                return self._read_chunked(bt, shape, dtype, cdims, filters)
        raise NotImplementedError(f"layout v{ver}")

    def _read_chunked(self, bt, shape, dtype, cdims, filters):
        out = np.zeros(shape, dtype)
        rank = len(shape)

        def walk(addr):
            n = self._rd(addr, 8 + 2 * self.O)
            assert n[:4] == b'TREE' and n[4] == 1
            level = n[5]; used = struct.unpack('<H', n[6:8])[0]
            ksz = 8 + 8 * (rank + 1)
            body = self._rd(addr + 8 + 2 * self.O, (used + 1) * ksz + used * self.O)
            q = 0
            for _ in range(used):
                k = body[q:q + ksz]; q += ksz
                child = self._u(body, q, self.O); q += self.O
                csize, fmask = struct.unpack('<II', k[:8])
                offs = [struct.unpack('<Q', k[8 + 8 * i:16 + 8 * i])[0] for i in range(rank)]
                if level > 0:
                    walk(child)
                else:
                    raw = self._rd(child, csize)
                    if 1 in filters and not (fmask & 1):
                        raw = zlib.decompress(raw)
                    blk = np.frombuffer(raw, dtype)[:int(np.prod(cdims))].reshape(cdims)
                    sl = tuple(slice(o, min(o + c, s)) for o, c, s in zip(offs, cdims, shape))
                    out[sl] = blk[tuple(slice(0, s.stop - s.start) for s in sl)]
        walk(bt)
        return out
