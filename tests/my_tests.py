import os

print(f"{os.getcwd()=}")

import sys
sys.path.append('./python')

import numpy as np
import needle as ndl
from needle import backend_ndarray as nd

def main():
    print(f"{os.getpid()=}")
    y = nd.NDArray(np.arange(16), device=nd.cpu()).reshape((4,4))
    z = y[1:3, 1:3]
    print(f"step 1: {y=}, {z=}")
    y[1:3, 1:3] = 0
    print(f"step 2: {y=}, {z=}")
    z = z.compact()
    y[1:3, 1:3] = 5
    print(f"step 3: {y=}, {z=}")

if __name__ == "__main__":
    main()
