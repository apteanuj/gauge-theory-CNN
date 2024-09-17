from netket.operator.spin import sigmax, sigmaz

from scipy import sparse as _sparse

from netket.utils.types import DType as _DType

from netket.hilbert import AbstractHilbert as _AbstractHilbert

from netket.operator._local_operator import LocalOperator as _LocalOperator


def id_mat(
    hilbert: _AbstractHilbert, site: int, dtype: _DType = float
) -> _LocalOperator:
    """
    Builds the identity matrix operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np

    N = hilbert.size_at_index(site)

    D = [1 for a in np.arange(1, N+1)]
    mat = np.diag(D)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)

# Clock and shift operators match with Q and P from 2008.00882.

def shift(
    hilbert: _AbstractHilbert, site: int, dtype: _DType = float
) -> _LocalOperator:
    """
    Builds the shift operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np

    N = hilbert.size_at_index(site)

    mat = np.zeros((N, N))
    mat[0, -1] = 1
    mat[1:, :-1] = np.eye(N - 1)
    mat = mat.transpose()
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def clock(
    hilbert: _AbstractHilbert, site: int, dtype: _DType = complex
) -> _LocalOperator:
    """
    Builds the clock operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    import netket.jax as nkjax

    if not nkjax.is_complex_dtype(dtype):
        import jax.numpy as jnp
        import warnings

        old_dtype = dtype
        dtype = jnp.promote_types(complex, old_dtype)
        warnings.warn(
            np.ComplexWarning(
                f"A complex dtype is required (dtype={old_dtype} specified). "
                f"Promoting to dtype={dtype}."
            )
        )

    N = hilbert.size_at_index(site)
    # Define the N-th root of unity
    omega = np.exp(2j * np.pi / N)

    mat = np.diag([omega**i for i in range(N)])
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def Hamiltonian_ZN(
    g, hilbert, plaq_ids, link_ids):
    """
    Builds ZN Hamiltonian given the hilbert space ids that make up the
    plaquettes and links
    """
    H = - (1 / (2 * g**2)) * sum([(- 2 * id_mat(hilbert=hilbert, site=i)*id_mat(hilbert=hilbert, site=j)*
            id_mat(hilbert=hilbert, site=k)*id_mat(hilbert=hilbert, site=l) + clock(hilbert=hilbert, site=i).H * clock(hilbert=hilbert, site=j).H *
         clock(hilbert=hilbert, site=k) * clock(hilbert=hilbert, site=l) + clock(hilbert=hilbert, site=l).H * clock(hilbert=hilbert, site=k).H *
         clock(hilbert=hilbert, site=j) * clock(hilbert=hilbert, site=i)) for (i,j,k,l) in plaq_ids])

    H = H - ((g**2) / 2) * sum([(- 2 * id_mat(hilbert=hilbert, site=i) + shift(hilbert=hilbert, site=i) + shift(hilbert=hilbert, site=i).H) for i in link_ids])

    return H

def Hamiltonian_ZN_electric(
    g, hilbert, plaq_ids, link_ids):
    """
    Builds electric part of ZN Hamiltonian given the hilbert space ids that make up the
    plaquettes and links
    """

    H_E = - ((g**2) / 2) * sum([(- 2 * id_mat(hilbert=hilbert, site=i) + shift(hilbert=hilbert, site=i) + shift(hilbert=hilbert, site=i).H) for i in link_ids])

    return H_E

def Hamiltonian_ZN_magnetic(
    g, hilbert, plaq_ids, link_ids):
    """
    Builds magnetic part of ZN Hamiltonian given the hilbert space ids that make up the
    plaquettes and links
    """
    H_B = - (1 / (2 * g**2)) * sum([(- 2 * id_mat(hilbert=hilbert, site=i)*id_mat(hilbert=hilbert, site=j)*
            id_mat(hilbert=hilbert, site=k)*id_mat(hilbert=hilbert, site=l) + clock(hilbert=hilbert, site=i).H * clock(hilbert=hilbert, site=j).H *
         clock(hilbert=hilbert, site=k) * clock(hilbert=hilbert, site=l) + clock(hilbert=hilbert, site=l).H * clock(hilbert=hilbert, site=k).H *
         clock(hilbert=hilbert, site=j) * clock(hilbert=hilbert, site=i)) for (i,j,k,l) in plaq_ids])

    return H_B


def gauge_ZN(
    hilbert, gauge_ids):
    """
    Builds ZN gauge operator and sums it for each lattice site
    Should return N_sites for gauge invariant state
    """
    theta = sum([shift(hilbert=hilbert, site=i)*shift(hilbert=hilbert, site=j)*(shift(hilbert=hilbert, site=k).H)*(shift(hilbert=hilbert, site=l).H) for (i,j,k,l) in gauge_ids])

    return theta

def gauge_zero_ZN(
    hilbert, gauge_ids):
    """
    Builds ZN gauge operator which is zero for gauge invariant state
    """
    theta_theta_dagger = sum([(shift(hilbert=hilbert, site=i)*shift(hilbert=hilbert, site=j)*(shift(hilbert=hilbert, site=k).H)*(shift(hilbert=hilbert, site=l).H)-id_mat(hilbert=hilbert, site=i)*id_mat(hilbert=hilbert, site=j)*
            id_mat(hilbert=hilbert, site=k)*id_mat(hilbert=hilbert, site=l))*(shift(hilbert=hilbert, site=l)*shift(hilbert=hilbert, site=k)*(shift(hilbert=hilbert, site=j).H)*(shift(hilbert=hilbert, site=i).H)-id_mat(hilbert=hilbert, site=i)*id_mat(hilbert=hilbert, site=j)*
            id_mat(hilbert=hilbert, site=k)*id_mat(hilbert=hilbert, site=l)) for (i,j,k,l) in gauge_ids])

    return theta_theta_dagger