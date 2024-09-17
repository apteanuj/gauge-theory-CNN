# numerical operations in the model should always use jax.numpy
# instead of numpy because jax supports computing derivatives.
# If you want to better understand the difference between the two, check
# https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html
import jax
import jax.numpy as jnp

# Flax is a framework to define models using jax
import flax

# we refer to `flax.linen` as `nn`. It's a repository of
# layers, initializers and nonlinear functions.
import flax.linen as nn

from typing import Any, Callable, Sequence
from jax import lax, random
from flax.core import freeze, unfreeze

# retrieve the Hilbert space id given the [x,y] coordinate
# and whether you want the x or y link attached to that site
def get_link_id(position, x_or_y, graph):
    if x_or_y == "x":
        return graph.id_from_position(position)
    if x_or_y == "y":
        return graph.id_from_position(position) + graph.n_nodes
    else:
        return "ERROR - need to specify x or y"

# One might think that removing the vmap's below and just using shifts etc.
# that act on the full batched object might be faster. But there is zero difference.
# Clearly jax is doing a great job of optimising everything.

# Function to convert a state, given by an array of integers in [0,..,N-1]
# to Z_N phases
# For what follows, we assume that the phases are directed such that the
# non-daggered phases are those on links leading away from the origin in positive x/y directions
@jax.jit
def states_to_phases(state, N):
    return jnp.exp(2 * jnp.pi * 1j * state / N)


# use vmap to batch generation of Z_N phases over an array of states
@jax.jit
def states_to_phases_batched(state_batched, N):
    return jax.vmap(states_to_phases, in_axes=(0,None), out_axes=0)(state_batched, N)


# function to convert a list of the link phase variables, indexed by the edge labels on the graph
# to an (2,nx,ny) tensor
# the first index labels whether you are picking the x or the y link
# the second and third index label the (x,y) coordinates of the associated node
# recall that are we labeling the Hilbert space so that the first N_sites elements
# are the x-links, and the next N_sites elements are the y-links
@jax.jit
def phases_to_tensor(links):
    x_links, y_links = jnp.split(links,2)
    xy_links = jnp.stack((x_links, y_links), axis=0)
    return jnp.reshape(xy_links,(2, nx, ny), order="F")


# function to convert a list of the link phase variables, indexed by the edge labels on the graph
# to an (dim,nx,ny) tensor
# the first index labels whether you are picking the x or the y links
# the second and third index label the (x,y) coordinates of the associated node
@jax.jit
def phases_to_tensor_batched(links_batched):
    return jax.vmap(phases_to_tensor, in_axes=(0), out_axes=0)(links_batched)


# calculate all plaquettes from the link phases
@jax.jit
def tensor_links_to_plaqs(tensor_links):
    # mu = xhat, nu = yhat
    # U_x,mu - select x links
    U_x_mu = tensor_links[0]
    # U_x,nu - select y links
    U_x_nu = tensor_links[1]
    # U_x+mu,nu
    U_x_mu_nu = jnp.roll(U_x_nu, (-1, 0), axis=(0,1))
    # U_x+nu,mu
    U_x_nu_mu = jnp.roll(U_x_mu, (0, -1), axis=(0,1))
    # plaq_x = U_x,mu * U_x+mu,nu * dag(U_x+nu,mu) * dag(U_x,nu)
    return jnp.multiply(jnp.multiply(U_x_mu, U_x_mu_nu), jnp.multiply(jnp.conj(U_x_nu_mu), jnp.conj(U_x_nu)))


# use vmap to batch generation of plaqs over an array of states
@jax.jit
def tensor_links_to_plaqs_batched(tensor_links_batched):
    return jax.vmap(tensor_links_to_plaqs, in_axes=(0), out_axes=0)(tensor_links_batched)


# function which takes in a tensor of plaquette variables and shifts them by a specified amount in x or y direction.
# In practice, this means it takes in W_x and returns W_(x + mu)
@jax.jit
def shift_plaqs(plaqs_tensor, mu_x, mu_y):
    return jnp.roll(plaqs_tensor, (-mu_x, -mu_y), axis=(0,1))


# only works for one *positive* unit of transport at the moment, as you need to multiply all the link variables!
# and the negative shifts are actually a little different!
@jax.jit
def transport_x(plaqs_tensor, links_tensor, mu_x):
    plaqs = shift_plaqs(plaqs_tensor, mu_x, 0)
    return jnp.multiply(jnp.multiply(links_tensor[0], plaqs), jnp.conj(links_tensor[0]))


# only works for one *positive* unit of transport at the moment, as you need to multiply all the link variables!
# and the negative shifts are actually a little different!
@jax.jit
def transport_y(plaqs_tensor, links_tensor, mu_y):
    plaqs = shift_plaqs(plaqs_tensor, 0, mu_y)
    return jnp.multiply(jnp.multiply(links_tensor[1], plaqs), jnp.conj(links_tensor[1]))


@jax.jit
def transport_x_batched(plaqs_tensor_batch, links_tensor_batch, mu_x):
    return jax.vmap(transport_x, in_axes=(0,0,None), out_axes=0)(plaqs_tensor_batch, links_tensor_batch, mu_x)


@jax.jit
def transport_y_batched(plaqs_tensor_batch, links_tensor_batch, mu_y):
    return jax.vmap(transport_y, in_axes=(0,0,None), out_axes=0)(plaqs_tensor_batch, links_tensor_batch, mu_y)


@jax.jit
def transport_x_batched_channel(W_tensor_batch_channel, links_tensor_batch, mu_x):
    return jax.vmap(transport_x_batched, in_axes=(1,None,None), out_axes=1)(W_tensor_batch_channel, links_tensor_batch, mu_x)


@jax.jit
def transport_y_batched_channel(W_tensor_batch_channel, links_tensor_batch, mu_y):
    return jax.vmap(transport_y_batched, in_axes=(1,None,None), out_axes=1)(W_tensor_batch_channel, links_tensor_batch, mu_y)

@jax.jit
def repeated_apply_with_initial_transport_x_batched_channel(carry, links_tensor_batch, mu_x):
    # The second argument is necessary for scan, even though we don't use it
    return transport_x_batched_channel(carry, links_tensor_batch, mu_x), carry

@jax.jit
def repeated_apply_transport_y_batched_channel(carry, links_tensor_batch, mu_y):
    next_carry = transport_y_batched_channel(carry, links_tensor_batch, mu_y)
    return next_carry, next_carry  # Return the next value as both the carry and the output


# standard (exact) solver for VMC_SRt - should be numerically more stable than iterative methods
linear_solver_exact = lambda A, b: jax.scipy.linalg.solve(A, b, assume_a="pos")


# function that initialises the vstate
def initialise_vstate(vstate, model, width, x, key, stddev_init):

    print("Initialising vstate via LSUV")

    # get initial weights, intermediate output of layers and the variances
    params_vstate = {'params': flax.core.copy(vstate.parameters, {})}
    intermediate_outputs = model.apply(params_vstate, x, return_intermediate=True)
    variances = {layer: jnp.sum(jnp.var(output, axis=0)).item() for layer, output in intermediate_outputs.items()}
    #print(variances)

    # save dictionary of shapes of layers and the parameters
    layer_shape_dict = jax.tree_util.tree_map(lambda x: x.shape, params_vstate)['params']
    layer_param_dict = params_vstate['params']

    # target variance of each layer should match the Plaq layer
    target_variance = variances['Plaq']
    target_variance = 1. / width 
    tolerance = 0.01 * target_variance

    # loop over each layer
    for layer_name in variances:

        # skip loop if first Plaq layer
        if layer_name == 'Plaq':
            continue

        for param_name, param_shape in layer_shape_dict[layer_name].items():
            # save dtype of weights
            param_dtype = layer_param_dict[layer_name][param_name].dtype
            std_found = False
            new_stddev = stddev_init

            # set bias to zero
            if param_name == 'bias':
                new_param = nn.initializers.normal(stddev=0., dtype=param_dtype)(key, param_shape)
                layer_param_dict[layer_name][param_name] = new_param
                continue

            # iteratively rescale weights until target variance reached
            # note that we don't do anything with the means here! they should be normalised to zero by a shift?
            iters = 0
            while not std_found:
                new_stddev = jnp.sqrt(target_variance).item() * new_stddev*(1./jnp.sqrt(variances[layer_name]).item())
                new_param = nn.initializers.normal(stddev=new_stddev, dtype=param_dtype)(key, param_shape)
                layer_param_dict[layer_name][param_name] = new_param
                new_params = {'params': layer_param_dict}
                intermediate_outputs = model.apply(new_params, x, return_intermediate=True)
                variances = {layer: jnp.sum(jnp.var(output, axis=0)).item() for layer, output in intermediate_outputs.items()}
                # print(variances)
                variance = variances[layer_name]
                if abs(variance - target_variance) < tolerance:
                    std_found = True

                # break if it takes more than 10 iterations
                iters = iters + 1
                if iters > 10:
                    std_found = True

    return layer_param_dict


# Helper function to load vstate
def load_vstate(vstate, dir: str, name: str):

    import os

    in_file = os.path.join(dir, name + ".mpack")

    with open(in_file, 'rb') as file:
        print("Opened vstate from " + in_file, end="\n\n")
        return flax.serialization.from_bytes(vstate, file.read())
        

# Helper function to save vstate
def save_vstate(vstate, dir: str, name: str):

    import os

    out_file = os.path.join(dir, name + ".mpack")

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    with open(out_file, "wb") as file:
        file.write(flax.serialization.to_bytes(vstate))

    print("Saved vstate to " + out_file, end="\n\n")


# Helper function to generate name from list of variable names
def get_name(arg_names, local_vars):
    """
    Concatenate specified variables into a single string, with variable names and values.

    Args:
        arg_names (list): List of variable names to include.
        local_vars (dict): Dictionary of local variables (usually from locals()).

    Returns:
        str: A string containing all specified arguments in the format "name(value)",
             joined by underscores. If the variable is 'g', its value is formatted to 4 decimal places.
    """
    def format_arg(name, value):
        if name == 'g' and isinstance(value, (int, float)):
            return f"{name}={value:.4f}"
        elif isinstance(value, str):
            return f"{name}='{value}'"
        else:
            return f"{name}={value}"

    args_list = [format_arg(var, local_vars[var]) for var in arg_names if var in local_vars]
    return '_'.join(args_list)


# write model/sampler/vstate params
def save_model_params(model, sampler, vstate, dir: str, name: str):

    import copy

    vstate_keys =  ['_n_discard_per_chain', '_chunk_size']

    vstate_dict = {k: vars(vstate)[k] for k in vstate_keys}

    #vstate_dict.update({'_n_samples': vars(vstate)['_chain_length'] * vars(sampler)['n_chains']})  # possible typo here 
    vstate_dict.update({'_n_samples': vars(vstate)['_chain_length'] * vars(sampler)['n_chains_per_rank']})  # possible typo here 
    vstate_dict = {key: repr(val) for key, val in vstate_dict.items()}

    model_dict = {key: repr(val) for key, val in vars(model).items()}
    # model_dict = copy.deepcopy(vars(model))
    # model_dict.pop('_id')
    # model_dict.pop('_state')

    sampler_dict = {key: repr(val) for key, val in vars(sampler).items()}
    # sampler_dict = copy.deepcopy(vars(sampler))

    params = {'model': model_dict, 'sampler': sampler_dict, 'vstate': vstate_dict}

    import os 

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    out_file = os.path.join(dir, name + ".txt")

    with open(out_file, 'w') as f:
        f.write(repr(params))

    print("Saved model/sampler/vstate params to " + out_file, end="\n\n")


# read model/sampler/vstate params
def load_model_params(dir: str, name: str):

    import os 
    import ast
    in_file = os.path.join(dir, name + ".txt")
    with open(in_file, 'r') as f:
        dict_str = f.read()
        # dict_from_file = ast.literal_eval(dict_str)
        dict_from_file = eval(dict_str)

    return dict_from_file


# write model/sampler/vstate params
def save_results(names, vars, dir: str, name: str):

    import os
    import csv

    out_file = os.path.join(dir, name + ".csv")
    file_exists = os.path.isfile(out_file)
    
    mode = 'a' if file_exists else 'w'

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    with open(out_file, mode, newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # Write the header row with variable names
            writer.writerow(names)
        
        # Write the data row
        writer.writerow(vars)
    
    print(f"{'Appended' if file_exists else 'Created file and saved'} results to {out_file}", end="\n")


# helper function to compute observable with variable number of samples
def compute_observables(obs_names: list, vstate, n_samples_obs, local_vars):

    # get original n_samples
    n_samples = vstate.n_samples

    # use new number of samples
    vstate.n_samples = n_samples_obs

    obs_expects = []

    for obs_name in obs_names:

        print("Computing <"+obs_name+">.")

        # compute the observable
        obs_expect = vstate.expect(local_vars[obs_name])

        print("<"+obs_name+">:", obs_expect, end="\n\n")

        obs_expects = obs_expects + [obs_expect]

    # reset n_samples
    vstate.n_samples = n_samples

    # return the stats for the observable
    return obs_expects


# helper function to computer tau_corr for a variable batch size
# return closest integer to correlation time
def get_tau_corr_int(vstate, O, n_samples_O):
    current_n_samples = vstate.n_samples
    vstate.n_samples = n_samples_O

    tau_corr_avg = vstate.expect(O).tau_corr

    vstate.n_samples = current_n_samples

    return round(tau_corr_avg.item())


# helper function to change sweep_size in a vstate when tau_corr gets large
def optimal_sweep_size(vstate, sampler, O, n_samples_O):

    # factor to multiply sweep_size by
    factor = get_tau_corr_int(vstate, O, n_samples_O)

    old_sweep_size = sampler.sweep_size

    new_sweep_size = factor * old_sweep_size

    print("Old sweep_size:", old_sweep_size)
    print("New sweep_size:", new_sweep_size)

    vstate.sampler = sampler.replace(sweep_size=new_sweep_size)

    print("Updated sampler with new sweep_size")

    return vstate