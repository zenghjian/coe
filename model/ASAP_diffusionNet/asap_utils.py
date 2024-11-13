# 3p
import torch


def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    scales = torch.sqrt(torch.sum(basis * massvec.unsqueeze(-1) * basis, dim=1))
    return torch.matmul(basisT, values * massvec.unsqueeze(-1)) / scales.unsqueeze(-1)


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        return cmatmul(ensure_complex(basis), ensure_complex(values))
    else:
        return torch.matmul(basis, values)


def cmatmul(A, B):

    A = ensure_complex(A)
    B = ensure_complex(B)

    return ensure_complex(
        torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag)
    ) + 1j * ensure_complex(torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real))


def cmatvecmul_stacked(mat, vec):
    # mat: (B,M,N,2)
    # vec: (B,N)
    # return: (B,M,2)
    return torch.stack(
        (torch.matmul(mat[..., 0], vec), torch.matmul(mat[..., 1], vec)), dim=-1
    )


def complex_dtype_equiv(d):
    if d == torch.float32:
        return torch.complex64
    elif d == torch.float64:
        return torch.complex128
    else:
        raise RuntimeError("unexpected type: " + str(d))


def ensure_complex(arr):
    if arr.is_complex():
        return arr
    return arr.to(complex_dtype_equiv(arr.dtype))