
#    signal_result_dir = args.result_dir + "/" + "signal_preprocessing"
#    if not path.exists(signal_result_dir):
#        os.makedirs(signal_result_dir)

import os

def get_signal_preprocessing_result_dir(args):
    """
    Get the directory for saving signal preprocessing results.
    """
    result_dir = os.path.join(args.result_dir, "signal_preprocessing")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def get_h_estimation_result_dir(args):
    """
    Get the directory for saving H estimation results.
    """
    result_dir = os.path.join(args.result_dir, "H_estimation")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def get_signal_reconstruction_result_dir(args):
    """
    Get the directory for saving signal reconstruction results.
    """
    result_dir = os.path.join(args.result_dir, "processed_phase")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir