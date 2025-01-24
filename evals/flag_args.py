import argparse


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def configure_as_flag_arg():
    """
    Argparse `store_true` is used for flag arguments whose boolean value simply matches the presence of the flag.

    Unfortunately, flag parameters are incompatible with how Marqtune (and SageMaker) take user-input as a
    dictionary of key/value pairs.

    To work around this, we use the following configuration to support parameters that are both flags and
    also accept explicit bool arguments.

    So, for example, the following command line arguments are equivalent:
    --frozen-right
    --frozen-right True

    By doing it this way, these parameters are compatible with existing training scripts that expect these
    parameters to remain flags as well as with Marqtune/SageMaker that can pass explicit bool arguments.
    """
    return {
        "type": _str2bool,
        "nargs": '?',
        "const": True,
    }
