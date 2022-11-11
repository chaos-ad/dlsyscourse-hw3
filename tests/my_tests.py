import os

print(f"{os.getcwd()=}")

import sys
sys.path.append('./python')

import numpy as np
import needle as ndl
import needle.backend_ndarray.ndarray as nd

def main():
    print(f"{os.getpid()=}")
    device = nd.cpu()
    matmul_dims = [
        # (2, 2, 2),
        # (1, 2, 3), 
        # (3, 4, 5), 
        # (5, 4, 3), 
        (8, 8, 8),
        # (16, 16, 16), 
        # (64, 64, 64), 
        # (72, 72, 72), 
        # (72, 73, 74), 
        # (74, 73, 72), 
        # (128, 128, 128)
    ]
    for (m, n, p) in matmul_dims:
        print(f"testing {(m, n, p)=}")
        _A = np.random.randn(m, n)
        _B = np.random.randn(n, p)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        nd_res = (A @ B).numpy()
        np_res = _A @ _B
        np.testing.assert_allclose(nd_res, np_res, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    main()
