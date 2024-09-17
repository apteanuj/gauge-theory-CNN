# numerical operations in the model should always use jax.numpy
# instead of numpy because jax supports computing derivatives.
# If you want to better understand the difference between the two, check
# https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html
import jax
import jax.numpy as jnp
import netket as nk

# Flax is a framework to define models using jax
import flax

# we refer to `flax.linen` as `nn`. It's a repository of
# layers, initializers and nonlinear functions.
import flax.linen as nn

from typing import Any, Callable, Sequence
from jax import lax, random
from flax.core import freeze, unfreeze
from ZN_functions import *


# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.

# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class ZN_to_Phase(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    N: int

    @nn.compact
    def __call__(self, x):

        phases_vector_batch = states_to_phases_batched(x, self.N)
        phases_tensor_batch = phases_to_tensor_batched(phases_vector_batch)

        # return the output
        return phases_tensor_batch


# The __call__(self, x) function should take as
# input a batch of link variables x.shape = (n_batch, 0 or 1, x coord, y coord)
# and should return a batch of plaqs of shape (n_batch, x coord, y coord)
class Plaq(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.

    @nn.compact
    def __call__(self, links):
        # compute plaqs from links
        plaqs = tensor_links_to_plaqs_batched(links)

        # convert links to a W_i variable by adding a channel index
        W_batch_i_x_y = jnp.expand_dims(plaqs, axis=1)

        # return the W_i variables with an extra channel
        return W_batch_i_x_y


# The __call__(self, x) function should take as
# input a batch of states x.shape = (n_samples, spin_states)
# and should return (n_samples, phases)
class LConvBilin(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    N_out: int
    param_dtype: jnp.dtype = jnp.complex64

    @nn.compact
    def __call__(self, links, plaqs_i):

        # input is batch of plaq variables with extra i index for one channel
        W_batch_i_x_y = plaqs_i

        # partial_repeated_apply_with_initial_transport_x_batched_channel = jax.tree_util.Partial(repeated_apply_with_initial_transport_x_batched_channel, links_tensor_batch=links, mu_x=1)
        
        # partial_repeated_apply_transport_y_batched_channel = jax.tree_util.Partial(repeated_apply_transport_y_batched_channel, links_tensor_batch=links, mu_y=1)
        
        # _, result_x = jax.lax.scan(partial_repeated_apply_with_initial_transport_x_batched_channel, W_batch_i_x_y, xs=None, length=self.shift+1)

        # _, result_y = jax.lax.scan(partial_repeated_apply_transport_y_batched_channel, W_batch_i_x_y, xs=None, length=self.shift)

        
        # consider only single positive shift along lattice axis, as in Vienna group sup. mat.
        W_batch_i_x_p_y = transport_x_batched_channel(W_batch_i_x_y, links, 1)
        # W_batch_i_x_m_y = transport_x_batched_channel(W_batch_i_x_y, links, -1)
        W_batch_i_x_y_p = transport_y_batched_channel(W_batch_i_x_y, links, 1)
        # W_batch_i_x_y_m = transport_y_batched_channel(W_batch_i_x_y, links, -1)

        # # combine the transported terms into a single object and then include the sum over k (the convolution)
        # # by folding it into the channel label
        
        W_trans_batch_i_x_y = jnp.concatenate([W_batch_i_x_y, W_batch_i_x_p_y, W_batch_i_x_y_p], axis=1)
        

        # include unit matrix at this point
        # without the unit matrix, you discard the smaller Wilson loops!
        batch_size = W_batch_i_x_y.shape[0]
        dim_x = W_batch_i_x_y.shape[2]
        dim_y = W_batch_i_x_y.shape[3]
        Id_batch_i_x_y = jnp.ones((batch_size,1,dim_x,dim_y), dtype=jnp.float32)

        # add conjugate of W variables to both W and W_trans
        # and the unit matrix
        # doubles the channels
        W_batch_i_x_y = jnp.concatenate([W_batch_i_x_y, jnp.conj(W_batch_i_x_y),Id_batch_i_x_y], axis=1)
        W_trans_batch_i_x_y = jnp.concatenate([W_trans_batch_i_x_y, jnp.conj(W_trans_batch_i_x_y), Id_batch_i_x_y], axis=1)

        # number of channels in W and W_trans with conjugates included
        W_channels = W_batch_i_x_y.shape[1]
        W_trans_channels = W_trans_batch_i_x_y.shape[1]

        alpha = self.param('alpha', nn.initializers.normal(), (self.N_out, W_channels, W_trans_channels), self.param_dtype)

        out = jnp.einsum('ijk, bjxy, bkxy -> bixy', alpha, W_batch_i_x_y, W_trans_batch_i_x_y, optimize='optimal')

        # return the output
        return out


# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states x.shape = (n_samples, spin_states)
# and should return (n_samples, phases)
# this allows for specifying a different shift
class LConvBilinShift(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    N_out: int
    shift: int = 1
    param_dtype: jnp.dtype = jnp.complex64
    stddev: Any = 0.01
    

    @nn.compact
    def __call__(self, links, plaqs_i):

        # input is batch of plaq variables with extra i index for one channel
        W_batch_i_x_y = plaqs_i

        # partial_repeated_apply_with_initial_transport_x_batched_channel = jax.tree_util.Partial(repeated_apply_with_initial_transport_x_batched_channel, links_tensor_batch=links, mu_x=1)
        
        # partial_repeated_apply_transport_y_batched_channel = jax.tree_util.Partial(repeated_apply_transport_y_batched_channel, links_tensor_batch=links, mu_y=1)
        
        # _, result_x = jax.lax.scan(partial_repeated_apply_with_initial_transport_x_batched_channel, W_batch_i_x_y, xs=None, length=self.shift+1)

        # _, result_y = jax.lax.scan(partial_repeated_apply_transport_y_batched_channel, W_batch_i_x_y, xs=None, length=self.shift)

        # store initial data in results list
        results = [W_batch_i_x_y]

        # store initial data in x variable which we'll overwrite
        x = W_batch_i_x_y

        # loop over the x shifts and append to results
        for i in range(self.shift):
            x = transport_x_batched_channel(x, links, 1)
            results.append(x)

        # store initial data in x variable which we'll overwrite
        x = W_batch_i_x_y

        # loop over the y shifts and append to results
        for i in range(self.shift):
            x = transport_y_batched_channel(x, links, 1)
            results.append(x)

        # combine along channel dimension
        W_trans_batch_i_x_y = jnp.concatenate(results, axis=1)

        # old version to shift by 1 only
        # # consider only positive shifts along lattice axis, as in Vienna group sup. mat.
        # W_batch_i_x_p_y = transport_x_batched_channel(W_batch_i_x_y, links, 1)
        # # W_batch_i_x_m_y = transport_x_batched_channel(W_batch_i_x_y, links, -1)
        # W_batch_i_x_y_p = transport_y_batched_channel(W_batch_i_x_y, links, 1)
        # # W_batch_i_x_y_m = transport_y_batched_channel(W_batch_i_x_y, links, -1)

        # # combine the transported terms into a single object and then include the sum over k (the convolution)
        # # by folding it into the channel label
        
        # W_trans_batch_i_x_y = jnp.concatenate([W_batch_i_x_y, W_batch_i_x_p_y, W_batch_i_x_y_p], axis=1)
        

        # include unit matrix at this point
        # without the unit matrix, you discard the smaller Wilson loops!
        batch_size = W_batch_i_x_y.shape[0]
        dim_x = W_batch_i_x_y.shape[2]
        dim_y = W_batch_i_x_y.shape[3]
        Id_batch_i_x_y = jnp.ones((batch_size,1,dim_x,dim_y), dtype=jnp.float32)

        # add conjugate of W variables to both W and W_trans
        # and the unit matrix
        # doubles the channels
        W_batch_i_x_y = jnp.concatenate([W_batch_i_x_y, jnp.conj(W_batch_i_x_y),Id_batch_i_x_y], axis=1)
        W_trans_batch_i_x_y = jnp.concatenate([W_trans_batch_i_x_y, jnp.conj(W_trans_batch_i_x_y), Id_batch_i_x_y], axis=1)

        # number of channels in W and W_trans with conjugates included
        W_channels = W_batch_i_x_y.shape[1]
        W_trans_channels = W_trans_batch_i_x_y.shape[1]

        alpha = self.param('alpha', nn.initializers.normal(stddev=self.stddev), (self.N_out, W_channels, W_trans_channels), self.param_dtype)

        out = jnp.einsum('ijk, bjxy, bkxy -> bixy', alpha, W_batch_i_x_y, W_trans_batch_i_x_y, optimize='optimal')

        # return the output
        return out


# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, spin_states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class MultiLConvBilin(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    # corresponds to N of Z_N
    N: int
    N_out: Sequence[int]
    features: Sequence[int]
    param_dtype: jnp.dtype = jnp.complex64

    @nn.compact
    def __call__(self, x):

        #if self.shifts == None:
        #    shifts = jnp.ones(len(self.N_out),int)
            
        links = ZN_to_Phase(self.N)(x)
        W_i_x_y = Plaq()(links)

        for i in range(len(self.N_out)):
            W_i_x_y = LConvBilin(N_out=self.N_out[i], param_dtype=self.param_dtype)(links, W_i_x_y)

        
        y = W_i_x_y.reshape(W_i_x_y.shape[0], -1)
        # y = jnp.reshape(W_i_x_y,(W_i_x_y.shape[0], -1))

        # these are the two that I removed
        # dense = nn.Dense(features=self.features * W_i_x_y_flat.shape[-1], kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)
        #dense = nn.Dense(features=self.features, kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)





        for i, feat in enumerate(self.features):
            y = nn.Dense(feat, kernel_init=nn.initializers.normal(), param_dtype=self.param_dtype)(y)
            y = nk.nn.reim_selu(y)
            


        # we apply the dense layer to the input
        #y = dense(W_i_x_y_flat)

        # the non-linearity is a simple ReLu
        #y = nn.relu(y)
        # return the output
        return jnp.sum(y, axis=-1)


# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, spin_states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class MultiLConvBilinShifts(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    # corresponds to N of Z_N
    N: int
    N_out: Sequence[int]
    features: Sequence[int]
    shifts: Sequence[int]
    param_dtype: jnp.dtype = jnp.complex64

    @nn.compact
    def __call__(self, x):

        #if self.shifts == None:
        #    shifts = jnp.ones(len(self.N_out),int)
            
        links = ZN_to_Phase(self.N)(x)
        W_i_x_y = Plaq()(links)

        for i in range(len(self.N_out)):
            W_i_x_y = LConvBilinShift(N_out=self.N_out[i], shift=self.shifts[i], param_dtype=self.param_dtype)(links, W_i_x_y)

        # these are the two that I removed
        y = W_i_x_y.reshape(W_i_x_y.shape[0], -1)

        # dense = nn.Dense(features=self.features * W_i_x_y_flat.shape[-1], kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)
        #dense = nn.Dense(features=self.features, kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)





        for i, feat in enumerate(self.features):
            y = nn.Dense(feat, kernel_init=nn.initializers.normal(), param_dtype=self.param_dtype)(y)
            y = nk.nn.reim_selu(y)
            


        # we apply the dense layer to the input
        #y = dense(W_i_x_y_flat)

        # the non-linearity is a simple ReLu
        #y = nn.relu(y)
        # return the output
        return jnp.sum(y, axis=-1)



# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, spin_states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class MultiLConvBilinShiftsVariance(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    # corresponds to N of Z_N
    N: int
    N_out: Sequence[int]
    features: Sequence[int]
    shifts: Sequence[int]
    precision: str = 'double'

    @nn.compact
    def __call__(self, x, return_intermediate=False):
        # determine dtype based on single or double precision
        # this allows us to use complex for the fully connected layers and real for the
        # LCB layers, while still changing the precision
        real_dtype = jnp.float32 if self.precision == 'single' else jnp.float64
        complex_dtype = jnp.complex64 if self.precision == 'single' else jnp.complex128

        outputs = {}

        #if self.shifts == None:
        #    shifts = jnp.ones(len(self.N_out),int)
            
        links = ZN_to_Phase(self.N)(x)
        W_i_x_y = Plaq()(links)

        N_sites_sqrt =  jnp.sqrt(W_i_x_y.shape[-1] * W_i_x_y.shape[-2])

        # divide by N_sites so that total variance of layer is 1
        W_i_x_y = W_i_x_y / N_sites_sqrt

        # add Plaq to outputs
        if return_intermediate:
            outputs['Plaq'] = W_i_x_y

        # ideally, want to use just real_dtype here, but VMC_SRt doesn't work with mixed dtypes yet!
        for i in range(len(self.N_out)):
            W_i_x_y = LConvBilinShift(N_out=self.N_out[i], shift=self.shifts[i], param_dtype=complex_dtype)(links, W_i_x_y)
            # W_i_x_y = nk.nn.reim_selu(W_i_x_y)

            # add LConvBilin to outputs
            if return_intermediate:
                outputs['LConvBilinShift_'+str(i)] = W_i_x_y

        

        # global average pooling
        # W_i_x_y has axes (batch, channel, x, y)
        # we average over the x and y coordinates
        # the output is then translation invariant, since the LCB is translation equivariant
        # output is (batch, channel)
        y = jnp.mean(W_i_x_y, axis=(-1, -2))

        # these are the two that I removed
        # used to just flatten all of the LCB output
        # but this won't be translation invariant!
        # you have to sum over (x,y) before you feed it into a dense layer
        # otherwise you will mix lattice sites with different weights
        # so the output won't be invariant!
        #y = W_i_x_y.reshape(W_i_x_y.shape[0], -1)

        for i, feat in enumerate(self.features):
            y = nn.Dense(feat, kernel_init=nn.initializers.normal(), param_dtype=complex_dtype)(y)
            y = nk.nn.reim_selu(y)

            #y = nk.nn.log_cosh(y)
            #y = nk.nn.reim(jnp.tanh)(y)

            # add Dense to outputs
            if return_intermediate:
                outputs['Dense_'+str(i)] = y

            
            
        #y = nk.nn.log_cosh(y)
            
        y = nn.Dense(1, kernel_init=nn.initializers.normal(), param_dtype=complex_dtype)(y)

        if return_intermediate:
            outputs['Dense_'+str(len(self.features))] = y
            
        y = jnp.sum(y, axis=-1)

        # y = nk.nn.log_cosh(y)

        # y_r = jnp.abs(y)
        # y_phi = jnp.angle(y)
        # y = jax.lax.complex(jnp.log(y_r), y_phi)
        
        # y_re = nn.activation.selu(jnp.real(y))
        # y_im = jnp.pi * nn.activation.soft_sign(jnp.imag(y))
        # y = jax.lax.complex(y_re, y_im)

        # y = jnp.log(y)

        
        if return_intermediate:
            return outputs
        else:
            return y

# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, spin_states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class MultiLConvBilinShiftsVarianceSymm(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    # corresponds to N of Z_N
    N: int
    N_out: Sequence[int]
    features: Sequence[int]
    shifts: Sequence[int]
    # translation symmetry of lattice to pass to GCNN - generate using graph.automorphisms()
    symmetries: Any
    symm_layers: int
    # output features for GCNN layer
    symm_features: int
    phase_only: bool = False
    precision: str = 'double'

    @nn.compact
    def __call__(self, x, return_intermediate=False):
        # determine dtype based on single or double precision
        # this allows us to use complex for the fully connected layers and real for the
        # LCB layers, while still changing the precision
        real_dtype = jnp.float32 if self.precision == 'single' else jnp.float64
        complex_dtype = jnp.complex64 if self.precision == 'single' else jnp.complex128

        outputs = {}

        #if self.shifts == None:
        #    shifts = jnp.ones(len(self.N_out),int)
            
        links = ZN_to_Phase(self.N)(x)
        W_i_x_y = Plaq()(links)

        N_sites_sqrt =  jnp.sqrt(W_i_x_y.shape[-1] * W_i_x_y.shape[-2])

        # divide by N_sites so that total variance of layer is 1
        W_i_x_y = W_i_x_y / N_sites_sqrt

        # add Plaq to outputs
        if return_intermediate:
            outputs['Plaq'] = W_i_x_y

        # ideally, want to use just real_dtype here, but VMC_SRt doesn't work with mixed dtypes yet!
        for i in range(len(self.N_out)):
            W_i_x_y = LConvBilinShift(N_out=self.N_out[i], shift=self.shifts[i], param_dtype=complex_dtype)(links, W_i_x_y)
            # W_i_x_y = nk.nn.reim_selu(W_i_x_y)

            # add LConvBilin to outputs
            if return_intermediate:
                outputs['LConvBilinShift_'+str(i)] = W_i_x_y

    

        # added these
        batch_size = W_i_x_y.shape[0]
        channel_size = W_i_x_y.shape[1]

        W_i_x_y = jnp.reshape(W_i_x_y,(batch_size, channel_size, -1), order="F")

    

        y = nk.models.GCNN(symmetries = self.symmetries, layers = self.symm_layers, features = self.symm_features, param_dtype=complex_dtype, equal_amplitudes=self.phase_only)(W_i_x_y)

        # W_i_x_y = nk.nn.DenseSymm(symmetries=self.symmetries,
        #                    features=self.symmetry_features,
        #                    kernel_init=nn.initializers.normal(),
        #                     param_dtype=complex_dtype)(W_i_x_y)


        # using log breaks the "equal_amplitudes" setting
        # y = jnp.log(y)

        
        if return_intermediate:
            return outputs
        else:
            return y

# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, spin_states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class MultiLConvBilinShiftsStd(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    # corresponds to N of Z_N
    N: int
    N_out: Sequence[int]
    features: Sequence[int]
    shifts: Sequence[int]
    stddevs: Sequence[Any]
    param_dtype: jnp.dtype = jnp.complex64
    

    @nn.compact
    def __call__(self, x):

        #if self.shifts == None:
        #    shifts = jnp.ones(len(self.N_out),int)
            
        links = ZN_to_Phase(self.N)(x)
        W_i_x_y = Plaq()(links)

        for i in range(len(self.N_out)):
            W_i_x_y = LConvBilinShift(N_out=self.N_out[i], shift=self.shifts[i], param_dtype=self.param_dtype, stddev=self.stddevs[i])(links, W_i_x_y)

        # these are the two that I removed
        y = W_i_x_y.reshape(W_i_x_y.shape[0], -1)

        # dense = nn.Dense(features=self.features * W_i_x_y_flat.shape[-1], kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)
        #dense = nn.Dense(features=self.features, kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)





        for i, feat in enumerate(self.features):
            y = nn.Dense(feat, kernel_init=nn.initializers.normal(), param_dtype=self.param_dtype)(y)
            y = nk.nn.reim_selu(y)
            


        # we apply the dense layer to the input
        #y = dense(W_i_x_y_flat)

        # the non-linearity is a simple ReLu
        #y = nn.relu(y)
        # return the output
        return jnp.sum(y, axis=-1)


# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, spin_states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class MultiLConvBilinSymm(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    # corresponds to N of Z_N
    N: int
    # output channels for equivariant layers
    N_out: Sequence[int]
    # width of dense layers
    features: Sequence[int]
    # translation symmetry of lattice to pass to GCNN - generate using graph.automorphisms()
    symmetries: Any
    # output features for GCNN layer
    symmetry_features: int

    @nn.compact
    def __call__(self, x):

        links = ZN_to_Phase(self.N)(x)
        W_i_x_y = Plaq()(links)

        for i in range(len(self.N_out)):
            W_i_x_y = LConvBilin(self.N_out[i])(links, W_i_x_y)

        # added these
        batch_size = W_i_x_y.shape[0]
        channel_size = W_i_x_y.shape[1]

        W_i_x_y_new = jnp.reshape(W_i_x_y,(batch_size, channel_size, -1), order="F")


        # these are the two that I removed
        #W_i_x_y_flat = W_i_x_y.reshape(W_i_x_y.shape[0], -1)
        #y = W_i_x_y_flat

        # dense = nn.Dense(features=self.features * W_i_x_y_flat.shape[-1], kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)
        #dense = nn.Dense(features=self.features, kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)



        y = nk.nn.DenseSymm(symmetries=self.symmetries,
                           features=self.symmetry_features,
                           kernel_init=nn.initializers.normal(),
                            param_dtype=jnp.complex64)(W_i_x_y_new)

        y = jnp.reshape(y, (batch_size, -1))

        for i, feat in enumerate(self.features):
            y = nn.Dense(feat, kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)(y)
            y = nk.nn.reim_selu(y)
            
            


        # we apply the dense layer to the input
        #y = dense(W_i_x_y_flat)

        # the non-linearity is a simple ReLu
        #y = nn.relu(y)
        # return the output
        return jnp.sum(y, axis=-1)


# A Flax model must be a class subclassing `nn.Module`
# The most compact way to define the model is this.
# The __call__(self, x) function should take as
# input a batch of states shape = (n_batch, spin_states_vector)
# and should return (n_batch, 0 or 1, x coord, y coord)
# where the coords refer to the node to which the links are associated
class MultiLConvBilin_No_ZN_to_Phase(nn.Module):

    # You can define attributes at the module-level
    # with a default. This allows you to easily change
    # some hyper-parameter without redefining the whole
    # flax module.
    # corresponds to N of Z_N
    N: int
    N_out: Sequence[int]
    features: Sequence[int]
    param_dtype: jnp.dtype = jnp.complex64

    @nn.compact
    def __call__(self, x):

        #if self.shifts == None:
        #    shifts = jnp.ones(len(self.N_out),int)
            
        links = x
        W_i_x_y = Plaq()(links)

        for i in range(len(self.N_out)):
            W_i_x_y = LConvBilin(N_out=self.N_out[i], param_dtype=self.param_dtype)(links, W_i_x_y)

        
        y = W_i_x_y.reshape(W_i_x_y.shape[0], -1)
        # y = jnp.reshape(W_i_x_y,(W_i_x_y.shape[0], -1))

        # these are the two that I removed
        # dense = nn.Dense(features=self.features * W_i_x_y_flat.shape[-1], kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)
        #dense = nn.Dense(features=self.features, kernel_init=nn.initializers.normal(), param_dtype=jnp.complex64)





        for i, feat in enumerate(self.features):
            y = nn.Dense(feat, kernel_init=nn.initializers.normal(), param_dtype=self.param_dtype)(y)
            y = nk.nn.reim_selu(y)
            


        # we apply the dense layer to the input
        #y = dense(W_i_x_y_flat)

        # the non-linearity is a simple ReLu
        #y = nn.relu(y)
        # return the output
        return jnp.sum(y, axis=-1)