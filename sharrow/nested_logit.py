from collections.abc import Mapping

import numba as nb
import numpy as np


@nb.njit(error_model="numpy", fastmath=True, cache=True)
def _utility_to_probability(
    n_alts,
    edges_up,  # int input shape=[edges]
    edges_dn,  # int input shape=[edges]
    mu_params,  # float input shape=[nests]
    start_slots,  # int input shape=[nests]
    len_slots,  # int input shape=[nests]
    only_utility,  #
    utility,  # float output shape=[nodes]
    logprob,  # float output shape=[nodes]
    probability,  # float output shape=[nodes]
):
    for up in range(n_alts, utility.size):
        up_nest = up - n_alts
        n_children_for_parent = len_slots[up_nest]
        shifter = -np.inf
        shifter_position = -1
        mu_up = mu_params[up_nest]
        if mu_up:
            for n in range(n_children_for_parent):
                edge = start_slots[up_nest] + n
                dn = edges_dn[edge]
                if utility[dn] > -np.inf:
                    z = utility[dn] / mu_up
                    if z > shifter:
                        shifter = z
                        shifter_position = dn
            for n in range(n_children_for_parent):
                edge = start_slots[up_nest] + n
                dn = edges_dn[edge]
                if utility[dn] > -np.inf:
                    if shifter_position == dn:
                        utility[up] += 1
                    else:
                        utility[up] += np.exp((utility[dn] / mu_up) - shifter)
            utility[up] = (np.log(utility[up]) + shifter) * mu_up
        else:  # mu_up is zero
            for n in range(n_children_for_parent):
                edge = start_slots[up_nest] + n
                dn = edges_dn[edge]
                if utility[dn] > utility[up]:
                    utility[up] = utility[dn]

    if only_utility:
        return

    for s in range(edges_up.size):
        dn = edges_dn[s]
        up = edges_up[s]
        mu_up = mu_params[up - n_alts]
        if np.isinf(utility[up]) and utility[up] < 0:
            logprob[dn] = -np.inf
        else:
            logprob[dn] = (utility[dn] - utility[up]) / mu_up

    # logprob becomes conditional_probability
    conditional_probability = logprob
    for i in range(logprob.size):
        conditional_probability[i] = np.exp(logprob[i])

    # probability
    probability[-1] = 1.0
    for s in range(edges_up.size - 1, -1, -1):
        dn = edges_dn[s]
        up = edges_up[s]
        probability[dn] = probability[up] * conditional_probability[dn]


@nb.njit
def _utility_to_probability_array(
    edges_up,  # int input shape=[edges]
    edges_dn,  # int input shape=[edges]
    mu_params,  # float input shape=[nests]
    start_slots,  # int input shape=[nests]
    len_slots,  # int input shape=[nests]
    utility,  # float output shape=[nodes]
):
    pr = np.zeros_like(utility)
    n_nodes = mu_params.size
    n_alts = utility.shape[1]
    for n in range(utility.shape[0]):
        _pr = np.zeros(n_nodes, dtype=np.float32)
        _util = np.zeros(n_nodes, dtype=np.float32)
        _logpr = np.zeros(n_nodes, dtype=np.float32)
        _util[:n_alts] = utility[n, :n_alts]
        _utility_to_probability(
            n_alts,
            edges_up,  # int input shape=[edges]
            edges_dn,  # int input shape=[edges]
            mu_params[n_alts:],  # float input shape=[nests]
            start_slots[n_alts:],  # int input shape=[nests]
            len_slots[n_alts:],  # int input shape=[nests]
            False,  #
            _util,  # float output shape=[nodes]
            _logpr,  # float output shape=[nodes]
            _pr,  # float output shape=[nodes]
        )
        pr[n, :] = _pr[:n_alts]
    return pr


@nb.njit
def _utility_to_logsums_array(
    edges_up,  # int input shape=[edges]
    edges_dn,  # int input shape=[edges]
    mu_params,  # float input shape=[nests]
    start_slots,  # int input shape=[nests]
    len_slots,  # int input shape=[nests]
    utility,  # float output shape=[nodes]
):
    logsums = np.zeros(utility.shape[0], dtype=np.float32)
    n_nodes = mu_params.size
    n_alts = utility.shape[1]
    for n in range(utility.shape[0]):
        _pr = np.zeros(n_nodes, dtype=np.float32)
        _util = np.zeros(n_nodes, dtype=np.float32)
        _logpr = np.zeros(n_nodes, dtype=np.float32)
        _util[:n_alts] = utility[n, :n_alts]
        _utility_to_probability(
            n_alts,
            edges_up,  # int input shape=[edges]
            edges_dn,  # int input shape=[edges]
            mu_params[n_alts:],  # float input shape=[nests]
            start_slots[n_alts:],  # int input shape=[nests]
            len_slots[n_alts:],  # int input shape=[nests]
            True,  #
            _util,  # float output shape=[nodes]
            _logpr,  # float output shape=[nodes]
            _pr,  # float output shape=[nodes]
        )
        logsums[n] = _util[-1]
    return logsums


def construct_nesting_tree(alternatives, nesting_settings):
    """
    Construct a larch NestingTree from ActivitySim settings.

    Parameters
    ----------
    alternatives : Mapping or Sequence
        If given as a Mapping (dict), the keys are the alternative names
        as strings, and the values are alternative code numbers to use
        in larch.  If given as a Sequence, the values are the alternative
        names, and unique sequential codes will be created starting from 1.
    nesting_settings : Mapping
        The 'NESTS' section of the ActivitySim config file.

    Returns
    -------
    NestingTree
    """
    try:
        from larch.model.tree import NestingTree
    except ImportError:
        raise ImportError("larch is required to construct nesting trees") from None

    if not isinstance(alternatives, Mapping):
        alt_names = list(alternatives)
        alt_codes = np.arange(1, len(alt_names) + 1)
        alternatives = dict(zip(alt_names, alt_codes))

    tree = NestingTree()
    nest_names_to_codes = alternatives.copy()
    nest_names_to_codes["root"] = 0
    for alt_name, alt_code in alternatives.items():
        tree.add_node(alt_code, name=alt_name)

    def make_nest(cfg, parent_code=0):
        nonlocal nest_names_to_codes
        if cfg["name"] != "root":
            if cfg["name"] not in nest_names_to_codes:
                n = tree.new_node(
                    name=cfg["name"],
                    parameter=str(cfg["coefficient"]),
                    parent=parent_code,
                )
                nest_names_to_codes[cfg["name"]] = n
            else:
                tree.add_edge(parent_code, nest_names_to_codes[cfg["name"]])
        for a in cfg["alternatives"]:
            if isinstance(a, str):
                tree.add_edge(nest_names_to_codes[cfg["name"]], nest_names_to_codes[a])
            else:
                make_nest(a, parent_code=nest_names_to_codes[cfg["name"]])

    make_nest(nesting_settings)

    return tree
