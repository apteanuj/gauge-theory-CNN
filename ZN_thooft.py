from netket.operator.spin import sigmax, sigmaz

from scipy import sparse as _sparse

from netket.utils.types import DType as _DType

from netket.hilbert import AbstractHilbert as _AbstractHilbert

from netket.operator._local_operator import LocalOperator as _LocalOperator

from operator import mul

from functools import reduce # python3 compatibility

from ZN_functions import get_link_id

from ZN_operators import clock, shift, id_mat

import jax.numpy as jnp

# Definition of custom t Hooft string operator for Z_N theory
def tHooftString(
    hilbert: _AbstractHilbert, graph, origin, x_length: int, y_length: int
):
    # if tHooft string is of zero length, return an identity matrix so
    # that the operator is trivial but one can still calculate an expectation value
    if x_length == 0 and y_length == 0:

        return id_mat(hilbert=hilbert, site=0)

    else:
        
        x_start = origin[0]
        y_start = origin[1]

        # get y link ids going in the +x direction
        id_list = [get_link_id([i,y_start],"y",graph) for i in range(x_start+1,x_start+x_length+1)]

        # get x link ids going in the +y direction
        id_list = id_list + [get_link_id([x_start+x_length,j],"x",graph) for j in range(y_start+1,y_start+y_length+1)]

        # make list of shift operators using link ids
        shift_list = [shift(hilbert=hilbert, site=i) for i in id_list]

        # use reduce to multiply together the list of shift operators
        return reduce(mul, shift_list, 1)


# sum of t Hooft string operators averaged over the lattice
# so the expectation value of this will be the lattice average of the t Hooft string
# note that you can't trust the variance calculation!
def tHooftString_lattice_average(
    hilbert: _AbstractHilbert, graph, x_length: int, y_length: int
):
    ny, nx = graph.extent

    tHooft_string = 0

    for x in range(0,nx):
        for y in range(0,ny):

            tHooft_string = tHooft_string + tHooftString(hilbert, graph, [x,y], x_length, y_length)

    tHooft_string = (1. / (nx * ny) ) * tHooft_string

    # use reduce to multiply together the list of shift operators
    return tHooft_string



# function to compute Binder cumulant of `t Hooft string
# this does not average over the lattice
def compute_tHooftString_Binder_cumulant(vstate, hilbert, graph, origin, x_length, y_length, n_samples = None, impose_reality = True):


    # check if number of samples is specified
    if n_samples is not None:

        # store n_samples for vstate
        n_samples_temp = vstate.n_samples

        # set n_samples to new value
        vstate.n_samples = n_samples


    print("Computing for", vstate.n_samples, "samples")


    W1 = tHooftString(hilbert, graph, origin, x_length, y_length)
    W2 = W1 @ W1
    W3 = W2 @ W1
    W4 = W2 @ W2

    # compute needed expectation values
    x1 = vstate.expect(W1).mean
    x2 = vstate.expect(W2).mean
    x3 = vstate.expect(W3).mean
    x4 = vstate.expect(W4).mean

    # impose that <W> should be real
    if impose_reality:
        x1 = jnp.real(x1)
        x2 = jnp.real(x2)
        x3 = jnp.real(x3)
        x4 = jnp.real(x4)

    # construct Binder cumulant
    BC = 6. * (x1**4) - 12. * (x1**2) * x2 + 3. * (x2**2) + 4. * x1 * x3 - x4
    BC = BC / (3. * (((x1**2) - x2)**2))


    # check if number of samples is specified
    if n_samples is not None:

        # restore n_samples for vstate
        vstate.n_samples = n_samples_temp


    return BC