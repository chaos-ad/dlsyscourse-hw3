import os

# print(f"{os.getcwd()=}")

import sys
sys.path.append('./python')

import numpy as np
import needle as ndl
import needle.backend_ndarray.ndarray as nd

def main():
    device = nd.cuda()
    lhs_shape, lhs_slices = ((3, 3, 3), (slice(1, 2, 1), slice(0, 1, 1), slice(0, 1, 1)))
    rhs_shape, rhs_slices = ((3, 3, 3), (slice(1, 2, 1), slice(0, 1, 1), slice(0, 1, 1)))
    _A = np.arange(nd.prod(lhs_shape)).reshape(*lhs_shape)
    _B = (np.arange(nd.prod(lhs_shape)) + 100).reshape(*rhs_shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    print(f"Before: {A=}")
    print(f"Before: {B=}")
    A[0, 0, 0] = B[0, 0, 0]
    print(f"After: {A=}")
    print(f"After: {B=}")
    _A[0, 0, 0] = _B[0, 0, 0]
    print(f"Numpy After: {_A=}")
    print(f"Numpy After: {_B=}")
    print(f"{np.isclose(A.numpy(), _A)=}")


if __name__ == "__main__":
    main()
