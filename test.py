#!/usr/bin/env python3
"""Simple fingerprint test using torch_fplib."""

import numpy as np
import torch_fplib
import time


def readvasp(vp):
    """Read VASP POSCAR file and return (lat, rxyz, types)."""
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())

    lat = np.array(buff[2:5], float)
    try:
        typt = np.array(buff[5], int)
    except ValueError:
        del buff[5]
        typt = np.array(buff[5], int)
    nat = sum(typt)
    pos = np.array(buff[7:7 + nat], float)
    types = []
    for i in range(len(typt)):
        types += [i + 1] * typt[i]
    types = np.array(types, int)
    rxyz = np.dot(pos, lat)
    return lat, rxyz, types


def test():
    znucl = [14, 8]
    lat1, rxyz1, types1 = readvasp('struct1.vasp')
    cell1 = (lat1, rxyz1, types1, znucl)

    t1 = time.time()
    fp = torch_fplib.get_lfp(cell1, cutoff=5.0, natx=100, orbital='s')
    t2 = time.time()
    print(f'Time: {t2 - t1:.4f} s')
    print(f'FP shape: {fp.shape}')
    print(f'FP[0]: {fp[0].numpy()[:10]}...')


if __name__ == "__main__":
    test()
