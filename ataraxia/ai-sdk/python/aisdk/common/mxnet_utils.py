import mxnet as mx


def load_checkpoint(sym_file, params_file):
    # pylint: disable=invalid-name
    """Load model checkpoint from file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        Epoch number of model we would like to load.

    Returns
    -------
    symbol : Symbol
        The symbol configuration of computation network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - Symbol will be loaded from ``prefix-symbol.json``.
    - Parameters will be loaded from ``prefix-epoch.params``.
    """
    symbol = mx.sym.load(sym_file)
    save_dict = mx.nd.load(params_file)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (symbol, arg_params, aux_params)
