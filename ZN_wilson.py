from netket.operator.spin import sigmax, sigmaz

from scipy import sparse as _sparse

from netket.utils.types import DType as _DType

from netket.hilbert import AbstractHilbert as _AbstractHilbert

from netket.operator._local_operator import LocalOperator as _LocalOperator

from operator import mul

from functools import reduce # python3 compatibility

from ZN_functions import get_link_id

from ZN_operators import clock, shift, id_mat


# Wilson string operator stretch between lattice sites in x-direction
def WilsonString(
    hilbert: _AbstractHilbert, graph, origin, x_length: int
):

    # if Wilson string is of zero length, return an identity matrix so
    # that the operator is triviswal but one can still calculate an expectation value
    if x_length == 0:

        return id_mat(hilbert=hilbert, site=0)

    else:

        x_start = origin[0]
        y_start = origin[1]

        # get x link ids going in the +x direction
        id_list = [get_link_id([i,y_start],"x",graph) for i in range(x_start,x_start+x_length)]

        # make list of shift operators using link ids
        clock_list = [clock(hilbert=hilbert, site=i) for i in id_list]

        # use reduce to multiply together the list of clock operators
        # not sure why I put an identity matrix here
        # return id_mat(hilbert=hi, site=get_link_id([x_y[0],x_y[1]],"x")) * reduce(mul, Wilson_list, 1)
        return reduce(mul, clock_list, 1)

# sum of Wilson string operators averaged over the lattice
# so the expectation value of this will be the lattice average of the t Hooft string
# note that you can't trust the variance calculation!
def WilsonString_lattice_average(
    hilbert: _AbstractHilbert, graph, x_length: int
):
    
    ny, nx = graph.extent

    Wilson_string = 0

    for x in range(0,nx):
        for y in range(0,ny):

            Wilson_string = Wilson_string + WilsonString(hilbert, graph, [x,y], x_length)

    Wilson_string = (1. / (nx * ny) ) * Wilson_string

    # use reduce to multiply together the list of shift operators
    return Wilson_string


# Wilson loop with bottom-left at origin=[x,y]
def WilsonLoop(
    hilbert: _AbstractHilbert, graph, origin, x_length: int, y_length: int):

    # if Wilson loop is of zero size, return an identity matrix so
    # that the operator is trivial but one can still calculate an expectation value
    if x_length == 0 and y_length == 0:
        return id_mat(hilbert=hilbert, site=0)

    else:

        x_start = origin[0]
        y_start = origin[1]

        right_ids = [get_link_id([x_start + i, y_start], "x", graph) for i in range(0, x_length)]
        up_ids = [get_link_id([x_start + x_length, y_start + j],"y", graph) for j in range(0, y_length)]
        left_ids = [get_link_id([x_start  + x_length - i, y_start + y_length],"x", graph) for i in range(1, x_length + 1)]
        down_ids = [get_link_id([x_start, y_start + y_length - j],"y", graph) for j in range(1, y_length + 1)]

        right_Qs = [clock(hilbert=hilbert, site=i) for i in right_ids]
        up_Qs = [clock(hilbert=hilbert, site=i) for i in up_ids]
        left_Qs = [clock(hilbert=hilbert, site=i).H for i in left_ids]
        down_Qs = [clock(hilbert=hilbert, site=i).H for i in down_ids]

        WLs = right_Qs + up_Qs + left_Qs + down_Qs

        return reduce(mul, WLs, 1)


# sum of Wilson string operators averaged over the lattice
# so the expectation value of this will be the lattice average of the t Hooft string
# note that you can't trust the variance calculation!
def WilsonLoop_lattice_average(
    hilbert: _AbstractHilbert, graph, x_length: int, y_length: int
):
    ny, nx = graph.extent

    Wilson_loop = 0

    for x in range(0,nx):
        for y in range(0,ny):

            Wilson_loop = Wilson_loop + WilsonLoop(hilbert, graph, [x,y], x_length, y_length)

    Wilson_loop = (1. / (nx * ny) ) * Wilson_loop

    # use reduce to multiply together the list of shift operators
    return Wilson_loop


# helper function to compute Creutz ratio at a single lattice site

def compute_Creutz_ratio(vstate, hilbert, graph, origin, length, n_samples = None, impose_reality = True):


    # check if number of samples is specified
    if n_samples is not None:

        # store n_samples for vstate
        n_samples_temp = vstate.n_samples

        # set n_samples to new value
        vstate.n_samples = n_samples


    if length < 2:
        raise ValueError("Need length of 2 or more, otherwise smaller loops are trivial!")

    # define Wilson loop operators W_lxl, W_(l-1)x(l-1), W_(l-1)xl, W_lx(l-1)
    W_l_l = WilsonLoop(hilbert, graph, origin, length, length)
    W_lm1_lm1 = WilsonLoop(hilbert, graph, origin, length-1, length-1)
    W_lm_l = WilsonLoop(hilbert, graph, origin, length-1, length)
    W_l_lm = WilsonLoop(hilbert, graph, origin, length, length-1)

    print("Computing for", vstate.n_samples, "samples")

    # compute expectation for largest loop and check its stats
    W_l_l = vstate.expect(W_l_l)
    W_lm1_lm1 = vstate.expect(W_lm1_lm1)
    W_lm_l = vstate.expect(W_lm_l)
    W_l_lm = vstate.expect(W_l_lm)


    # check if number of samples is specified
    if n_samples is not None:

        # restore n_samples for vstate
        vstate.n_samples = n_samples_temp


    print(length,"x",length,"loop stats:", W_l_l)
    print(length-1,"x",length-1,"loop stats:", W_lm1_lm1)
    print(length-1,"x",length,"loop stats:", W_lm_l)
    print(length,"x",length-1,"loop stats:", W_l_lm)

    # get error of mean for each
    delta_W_l_l = W_l_l.error_of_mean
    delta_W_lm1_lm1 = W_lm1_lm1.error_of_mean
    delta_W_lm_l = W_lm_l.error_of_mean
    delta_W_l_lm = W_l_lm.error_of_mean

    if impose_reality:
        print("Imposing reality")
        # compute expectations and take real part for remaining loops
        W_l_l = jnp.real(W_l_l.mean)
        W_lm1_lm1 = jnp.real(W_lm1_lm1.mean)
        W_lm_l = jnp.real(W_lm_l.mean)
        W_l_lm = jnp.real(W_l_lm.mean)

    else:
        print("Without imposing reality")
        # compute expectations remaining loops
        W_l_l = W_l_l.mean
        W_lm1_lm1 = W_lm1_lm1.mean
        W_lm_l = W_lm_l.mean
        W_l_lm = W_l_lm.mean


    creutz = jnp.log((W_lm_l * W_l_lm) / (W_l_l * W_lm1_lm1))

    error = jnp.sqrt((delta_W_l_l/W_l_l)**2 + (delta_W_lm1_lm1/W_lm1_lm1)**2 + (delta_W_lm_l/W_lm_l)**2 + (delta_W_l_lm/W_l_lm)**2)


    return creutz, error



# helper function to compute Creutz ratio at a single lattice site

def compute_Creutz_ratio_lattice_average(vstate, hilbert, graph, length, n_samples = None, impose_reality = True):


    # check if number of samples is specified
    if n_samples is not None:

        # store n_samples for vstate
        n_samples_temp = vstate.n_samples

        # set n_samples to new value
        vstate.n_samples = n_samples


    if length < 2:
        raise ValueError("Need length of 2 or more, otherwise smaller loops are trivial!")

    # define Wilson loop operators W_lxl, W_(l-1)x(l-1), W_(l-1)xl, W_lx(l-1)
    W_l_l = WilsonLoop_lattice_average(hilbert, graph, length, length)
    W_lm1_lm1 = WilsonLoop_lattice_average(hilbert, graph, length-1, length-1)
    W_lm_l = WilsonLoop_lattice_average(hilbert, graph, length-1, length)
    W_l_lm = WilsonLoop_lattice_average(hilbert, graph, length, length-1)

    print("Computing for", vstate.n_samples, "samples")

    # compute expectation for largest loop and check its stats
    W_l_l = vstate.expect(W_l_l)
    W_lm1_lm1 = vstate.expect(W_lm1_lm1)
    W_lm_l = vstate.expect(W_lm_l)
    W_l_lm = vstate.expect(W_l_lm)


    # check if number of samples is specified
    if n_samples is not None:

        # restore n_samples for vstate
        vstate.n_samples = n_samples_temp


    print(length,"x",length,"loop stats:", W_l_l)
    print(length-1,"x",length-1,"loop stats:", W_lm1_lm1)
    print(length-1,"x",length,"loop stats:", W_lm_l)
    print(length,"x",length-1,"loop stats:", W_l_lm)

    # get error of mean for each
    delta_W_l_l = W_l_l.error_of_mean
    delta_W_lm1_lm1 = W_lm1_lm1.error_of_mean
    delta_W_lm_l = W_lm_l.error_of_mean
    delta_W_l_lm = W_l_lm.error_of_mean

    if impose_reality:
        # compute expectations and take real part for remaining loops
        print("Imposing reality")
        W_l_l = jnp.real(W_l_l.mean)
        W_lm1_lm1 = jnp.real(W_lm1_lm1.mean)
        W_lm_l = jnp.real(W_lm_l.mean)
        W_l_lm = jnp.real(W_l_lm.mean)

    else:
        # compute expectations remaining loops
        print("Without imposing reality")
        W_l_l = W_l_l.mean
        W_lm1_lm1 = W_lm1_lm1.mean
        W_lm_l = W_lm_l.mean
        W_l_lm = W_l_lm.mean

    error = jnp.sqrt((delta_W_l_l/W_l_l)**2 + (delta_W_lm1_lm1/W_lm1_lm1)**2 + (delta_W_lm_l/W_lm_l)**2 + (delta_W_l_lm/W_l_lm)**2)

    print("Approximate absolute error:", error)

    creutz = jnp.log((W_lm_l * W_l_lm) / (W_l_l * W_lm1_lm1))

    return creutz, error


# function to compute Binder cumulant of lxl Wilson loop
# this does not average over the lattice
def compute_WilsonLoop_Binder_cumulant(vstate, hilbert, graph, origin, length, ZN, n_samples = None, impose_reality = True):


    # check if number of samples is specified
    if n_samples is not None:

        # store n_samples for vstate
        n_samples_temp = vstate.n_samples

        # set n_samples to new value
        vstate.n_samples = n_samples


    print("Computing for", vstate.n_samples, "samples")
    

    # since we compute up to <O^4>, we can simplify for ZN < 5
    # get ZN from dimension of Hilbert space at first site
    # ZN = hilbert.size_at_index(0)

    if ZN==2:

        # W^2 = 1
        W1 = WilsonLoop(hilbert, graph, origin, x_length=length, y_length=length)

        # compute needed expectation values
        x1 = vstate.expect(W1)

        print("<W> loop stats:", x1)

        x1 = x1.mean

        # impose that <W> should be real
        if impose_reality:
            print("Imposing reality")
            x1 = jnp.real(x1)

        # construct Binder cumulant
        BC = (2. - 6. * (x1**2)) / (3. - 3. * (x1**2))

    elif ZN==3:

        # W^3 = 1, W^2 = W^\dagger, but <W> should be real, so only depends
        # on <W> again!
        W1 = WilsonLoop(hilbert, graph, origin, x_length=length, y_length=length)

        # compute needed expectation values
        x1 = vstate.expect(W1)

        print("<W> loop stats:", x1)

        x1 = x1.mean

        # impose that <W> should be real
        if impose_reality:
            print("Imposing reality")
            x1 = jnp.real(x1)

        # construct Binder cumulant
        BC = 2. + 1. / (x1 - (x1**2))

    elif ZN==4:

        # W^4 = 1, W^3 = W^\dagger, but <W> should be real, so depends
        # on <W> and <W^2>
        W1 = WilsonLoop(hilbert, graph, origin, x_length=length, y_length=length)
        W2 = W1 @ W1

        # compute needed expectation values
        x1 = vstate.expect(W1)
        x2 = vstate.expect(W2)

        print("<W> loop stats:", x1)
        print("<W^2> loop stats:", x2)

        x1 = x1.mean
        x2 = x2.mean

        # impose that <W> should be real
        if impose_reality:
            print("Imposing reality")
            x1 = jnp.real(x1)
            x2 = jnp.real(x2)

        # construct Binder cumulant
        BC = -1. + 6. * (x1**4) + (x1**2) * (4. - 12. * x2) + 3. * (x2**2)
        BC = BC / (3. * (((x1**2) - x2)**2))

    else:

        W1 = WilsonLoop(hilbert, graph, origin, x_length=length, y_length=length)
        W2 = W1 @ W1
        W3 = W2 @ W1
        W4 = W2 @ W2

        # compute needed expectation values
        x1 = vstate.expect(W1)
        x2 = vstate.expect(W2)
        x3 = vstate.expect(W3)
        x4 = vstate.expect(W4)

        print("<W> loop stats:", x1)
        print("<W^2> loop stats:", x2)
        print("<W^3> loop stats:", x3)
        print("<W^4> loop stats:", x4)

        x1 = x1.mean
        x2 = x2.mean
        x3 = x3.mean
        x4 = x4.mean

        # impose that <W> should be real
        if impose_reality:
            print("Imposing reality")
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





# functions to fetch the Hilbert space ids for the links that come with a Q or a Qdagger in a Wilson loop
# specify the graph of the lattice, the bottom left origin of the Wilson loop, and the x/y lengths
# return a list of Hilbert space id's
import operator

def pos_link_ids_in_loop(graph, origin, x_length, y_length):
    
    # positions of positive x-links starting from origin = [i,j] site
    x_i_j_positions = [list(map(operator.add, origin, [i, 0])) for i in range(x_length)]
    # positions of positive y-links starting from [i+x_length,j] site
    y_ip_j_positions = [list(map(operator.add, origin, [x_length, j])) for j in range(y_length)]
    
    # ids of the above links
    x_i_j_ids = graph.id_from_position(x_i_j_positions)
    y_ip_j_ids = graph.id_from_position(y_ip_j_positions) + graph.n_nodes
    
    return jnp.concatenate((x_i_j_ids, y_ip_j_ids))

def neg_link_ids_in_loop(graph, origin, x_length, y_length):
    
    # positions of negative x-links starting from [i+x_length,j+y_length] site
    x_im_j_positions = [list(map(operator.add, origin, [-i+x_length-1, y_length])) for i in range(x_length)]
    # positions of negative y-links starting from [i,j+y_length] site
    y_i_jm_positions = [list(map(operator.add, origin, [0, -j+y_length-1])) for j in range(y_length)]
    
    x_im_j_ids = graph.id_from_position(x_im_j_positions)
    y_i_jm_ids = graph.id_from_position(y_i_jm_positions) + graph.n_nodes
    
    return jnp.concatenate((x_im_j_ids, y_i_jm_ids))


# Definition of custom Wilson loop operator for Z_N theory

import netket as nk
from netket.operator import AbstractOperator

class WilsonLoopOperator(AbstractOperator):

    def __init__(self, graph, hi, origin, x_length: int, y_length: int, power=1):
        super().__init__(hi)
        self._graph = graph
        self._origin = origin
        self._x_length = x_length
        self._y_length = y_length
        self._pos_links = pos_link_ids_in_loop(graph, origin, x_length, y_length)
        self._neg_links = neg_link_ids_in_loop(graph, origin, x_length, y_length)
        self._power = power

    @property
    def dtype(self):
        return complex

    @property
    def origin(self):
        """Origin of Wilson loop (bottom left lattice site)"""
        return self._origin

    @property
    def x_length(self) -> int:
        """Number of lattice sites in x direction"""
        return self._x_length

    @property
    def y_length(self) -> int:
        """Number of lattice sites in y direction"""
        return self._y_length

    @property
    def pos_links(self):
        """Links that come with standard clock operator"""
        return self._pos_links

    @property
    def neg_links(self):
        """Links that come with daggered clock operator"""
        return self._neg_links

    @property
    def power(self):
        """Power of Wilson loop operator"""
        return self._power

    @property
    def graph(self):
        """Graph that operator acts on"""
        return self._graph

    def __matmul__(self, other):
        """Define multiplication for identical Wilson loop operators"""
        assert self.graph == other.graph, "Graphs different"
        assert self.hilbert == other.hilbert, "Hilbert spaces different"
        assert self.origin == other.origin, "Wilson loop origins different"
        assert self.x_length == other.x_length, "x-length different"
        assert self.y_length == other.y_length, "y-length different"
        # return Wilson Loop operator with identitical properties but add the powers
        # use self.__class__ so that the RealWilsonLoopOperator which inherits from this class also returns a RealWilsonLoopOperator
        return self.__class__(self.graph, self.hilbert, self.origin, self.x_length, self.y_length, power = (self.power + other.power))

class RealWilsonLoopOperator(WilsonLoopOperator):
    pass

from functools import partial # partial(sum, axis=1)(x) == sum(x, axis=1)
import jax
import jax.numpy as jnp

@partial(jax.vmap, in_axes=(0, None))
def get_conns_and_mels(sigma, op):
    # this code only works if sigma is a single bitstring
    assert sigma.ndim == 1

    #origin = op.origin
    #x_length = op.x_length
    #y_length = op.y_length

    pos_links = op.pos_links
    neg_links = op.neg_links

    # N of Z_N can be extracted from the dimension of the Hilbert space of the operator at a single site
    ZN = op.hilbert.shape[0]

    links = (-sigma + (ZN-1.)) / 2
    
    #print("links:", links)

    pos_link_vars = links[pos_links]
    neg_link_vars = links[neg_links]
    #print("pos_link_vars:", pos_link_vars)
    #print("neg_link_vars:", neg_link_vars)

    pos_phase_vars = jnp.exp(2 * jnp.pi * 1j * pos_link_vars / ZN)
    neg_phase_vars = jnp.exp(-2 * jnp.pi * 1j * neg_link_vars / ZN)
    #print("pos_phase_vars:", pos_phase_vars)
    #print("neg_phase_vars:", neg_phase_vars)

    # if we're taking a power of the Wilson loop operator, take the appropriate power of the phase
    total_phase = (jnp.prod(pos_phase_vars) * jnp.prod(neg_phase_vars)) ** op.power

    
    #print("phase:", total_phase)

    # get number of spins
    N = sigma.shape[-1]
    # repeat eta 1 times
    eta = jnp.tile(sigma, (1,1))
    # diagonal indices
    #ids = jnp.diag_indices(N)
    # flip those indices
    #eta = eta.at[ids].set(-eta.at[ids].get())
    return eta, jnp.array([total_phase])

@partial(jax.vmap, in_axes=(None, None, 0,0,0))
def e_loc(logpsi, pars, sigma, eta, mels):
    return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

@nk.vqs.expect.dispatch
def expect(vs: nk.vqs.MCState, op: WilsonLoopOperator):
    return _expect(vs._apply_fun, vs.variables, vs.samples, op)

@partial(jax.jit, static_argnums=(0,3))
def _expect(logpsi, variables, sigma, op):
    n_chains = sigma.shape[-2]
    N = sigma.shape[-1]
    # flatten all batches
    sigma = sigma.reshape(-1, N)

    eta, mels = get_conns_and_mels(sigma, op)

    E_loc = e_loc(logpsi, variables, sigma, eta, mels)

    # reshape back into chains to compute statistical information
    E_loc = E_loc.reshape(-1, n_chains)

    # this function computes things like variance and convergence information.
    return nk.stats.statistics(E_loc)


# define new functions for evaluating <re W>

@partial(jax.vmap, in_axes=(0, None))
def get_conns_and_mels_real(sigma, op):
    # this code only works if sigma is a single bitstring
    assert sigma.ndim == 1

    #origin = op.origin
    #x_length = op.x_length
    #y_length = op.y_length

    pos_links = op.pos_links
    neg_links = op.neg_links

    # N of Z_N can be extracted from the dimension of the Hilbert space of the operator at a single site
    ZN = op.hilbert.shape[0]

    links = (-sigma + (ZN-1.)) / 2
    
    #print("links:", links)

    pos_link_vars = links[pos_links]
    neg_link_vars = links[neg_links]
    #print("pos_link_vars:", pos_link_vars)
    #print("neg_link_vars:", neg_link_vars)

    pos_phase_vars = jnp.exp(2 * jnp.pi * 1j * pos_link_vars / ZN)
    neg_phase_vars = jnp.exp(-2 * jnp.pi * 1j * neg_link_vars / ZN)
    #print("pos_phase_vars:", pos_phase_vars)
    #print("neg_phase_vars:", neg_phase_vars)

    # find the phase
    total_phase = (jnp.prod(pos_phase_vars) * jnp.prod(neg_phase_vars))

    # if we're taking a power of the Wilson loop operator, take the appropriate power of the phase
    total_phase = total_phase ** op.power

    # include only the real part of the phase
    total_phase = jnp.real(total_phase)



    
    #print("phase:", total_phase)

    # get number of spins
    N = sigma.shape[-1]
    # repeat eta 1 times
    eta = jnp.tile(sigma, (1,1))
    # diagonal indices
    #ids = jnp.diag_indices(N)
    # flip those indices
    #eta = eta.at[ids].set(-eta.at[ids].get())
    return eta, jnp.array([total_phase])
    
@nk.vqs.expect.dispatch
def expect(vs: nk.vqs.MCState, op: RealWilsonLoopOperator):
    return _expect_real(vs._apply_fun, vs.variables, vs.samples, op)

@partial(jax.jit, static_argnums=(0,3))
def _expect_real(logpsi, variables, sigma, op):
    n_chains = sigma.shape[-2]
    N = sigma.shape[-1]
    # flatten all batches
    sigma = sigma.reshape(-1, N)

    eta, mels = get_conns_and_mels_real(sigma, op)

    E_loc = e_loc(logpsi, variables, sigma, eta, mels)

    # reshape back into chains to compute statistical information
    E_loc = E_loc.reshape(-1, n_chains)

    # this function computes things like variance and convergence information.
    return nk.stats.statistics(E_loc)
    
    

# custom operator for a pair of Wilson loops
class PairWilsonLoopOperator(AbstractOperator):

    def __init__(self, graph, hi, origin_1, x_length_1: int, y_length_1: int, origin_2, x_length_2: int, y_length_2: int):
        super().__init__(hi)
        self._origin_1 = origin_1
        self._x_length_1 = x_length_1
        self._y_length_1 = y_length_1
        self._origin_2 = origin_2
        self._x_length_2 = x_length_2
        self._y_length_2 = y_length_2
        # get undaggered and daggered links for each Wilson loop
        self._pos_links_1 = pos_link_ids_in_loop(graph, origin_1, x_length_1, y_length_1)
        self._neg_links_1 = neg_link_ids_in_loop(graph, origin_1, x_length_1, y_length_1)
        self._pos_links_2 = pos_link_ids_in_loop(graph, origin_2, x_length_2, y_length_2)
        self._neg_links_2 = neg_link_ids_in_loop(graph, origin_2, x_length_2, y_length_2)
        # add together lists to have single list of the undaggered and daggered links
        # this will need changing when moving to a non-abelian example
        self._pos_links = jnp.concatenate((self._pos_links_1, self._pos_links_2))
        self._neg_links = jnp.concatenate((self._neg_links_1, self._neg_links_2))

    @property
    def dtype(self):
        return complex

    @property
    def pos_links(self):
        """Number of lattice sites in y direction"""
        return self._pos_links

    @property
    def neg_links(self):
        """Number of lattice sites in y direction"""
        return self._neg_links

from functools import partial # partial(sum, axis=1)(x) == sum(x, axis=1)
import jax
import jax.numpy as jnp

@partial(jax.vmap, in_axes=(0, None))
def get_conns_and_mels_pair(sigma, op):
    # this code only works if sigma is a single bitstring
    assert sigma.ndim == 1

    #origin = op.origin
    #x_length = op.x_length
    #y_length = op.y_length

    pos_links = op.pos_links
    neg_links = op.neg_links

    # N of Z_N can be extracted from the dimension of the Hilbert space of the operator at a single site
    ZN = op.hilbert.shape[0]

    # conversion of states to links
    links = (-sigma + (ZN-1.)) / 2
    #print("links:", links)

    pos_link_vars = links[pos_links]
    neg_link_vars = links[neg_links]
    #print("pos_link_vars:", pos_link_vars)
    #print("neg_link_vars:", neg_link_vars)

    pos_phase_vars = jnp.exp(2 * jnp.pi * 1j * pos_link_vars / ZN)
    neg_phase_vars = jnp.exp(-2 * jnp.pi * 1j * neg_link_vars / ZN)
    #print("pos_phase_vars:", pos_phase_vars)
    #print("neg_phase_vars:", neg_phase_vars)

    total_phase = jnp.prod(pos_phase_vars) * jnp.prod(neg_phase_vars)
    #print("phase:", total_phase)

    # get number of spins
    N = sigma.shape[-1]
    # repeat eta 1 times
    eta = jnp.tile(sigma, (1,1))
    # diagonal indices
    #ids = jnp.diag_indices(N)
    # flip those indices
    #eta = eta.at[ids].set(-eta.at[ids].get())
    return eta, jnp.array([total_phase])

@partial(jax.vmap, in_axes=(None, None, 0,0,0))
def e_loc_pair(logpsi, pars, sigma, eta, mels):
    return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

@nk.vqs.expect.dispatch
def expect(vs: nk.vqs.MCState, op: PairWilsonLoopOperator):
    return _expect_pair(vs._apply_fun, vs.variables, vs.samples, op)

@partial(jax.jit, static_argnums=(0,3))
def _expect_pair(logpsi, variables, sigma, op):
    n_chains = sigma.shape[-2]
    N = sigma.shape[-1]
    # flatten all batches
    sigma = sigma.reshape(-1, N)

    eta, mels = get_conns_and_mels_pair(sigma, op)

    E_loc = e_loc_pair(logpsi, variables, sigma, eta, mels)

    # reshape back into chains to compute statistical information
    E_loc = E_loc.reshape(-1, n_chains)

    # this function computes things like variance and convergence information.
    return nk.stats.statistics(E_loc)

# input is the variational state, the hamiltonian, and the "zero energy shift" of the Hamiltonian
# for example, for our Z_N hamiltonian, it contains a constant shift of the ground state energy by 
# L^2 (g^2 + g^-2)
def vscore(vstate, hamiltonian, zero_shift):
    import numpy as np
    # number of links
    N_links = hamiltonian.hilbert.size
    stats = vstate.expect(hamiltonian)
    mean = stats.mean
    variance = stats.variance
    return np.real((N_links * variance) / (mean - zero_shift)**2)

# input is the variational state, the hamiltonian, and the "zero energy shift" of the Hamiltonian
# for example, for our Z_N hamiltonian, it contains a constant shift of the ground state energy by 
# L^2 (g^2 + g^-2)
# this looks basically the same as computing mean / sqrt( var / N)
def StN(stats):
    import numpy as np
    mean = stats.mean
    error = stats.error_of_mean
    return np.abs(mean) / error
    