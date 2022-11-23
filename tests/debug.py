import os

# print(f"{os.getcwd()=}")

import sys
sys.path.append('./python')

import numpy as np
import needle as ndl
import needle.backend_ndarray.ndarray as nd

def main():
    A_dims = (128, 128)
    device = nd.cuda()
    A = nd.array(np.arange(nd.prod(A_dims)).reshape(*A_dims), device=device)
    print(f"{A.shape=}")
    # print(f"{A=}")

    B_dims = (128, 128)
    B = nd.array(np.arange(nd.prod(B_dims)).reshape(*B_dims), device=device)
    print(f"{B.shape=}")
    # print(f"{B=}")

    C = A @ B
    print(f"{C.shape=}")
    # print(f"{C=}")

    D = A.numpy() @ B.numpy()
    print(f"{D.shape=}")
    # print(f"{D=}")

    np.testing.assert_allclose(C.numpy(), D, rtol=1e-5, atol=1e-5)

    
    # _A = np.random.randn(m, n)
    # _B = np.random.randn(n, p)
    # A = nd.array(_A, device=device)
    # B = nd.array(_B, device=device)
    # np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    main()
