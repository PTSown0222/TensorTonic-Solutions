import numpy as np
def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    val_arr = np.array(values)
    val_min = np.min(val_arr)
    val_max = np.max(val_arr)

    if val_max == val_min:
        return [0] * len(val_arr)

    w = (val_max - val_min) / num_bins
    bin_x = np.floor((val_arr - val_min) / w)
    bin_x = np.minimum(bin_x, num_bins - 1)
    return bin_x.astype(int).tolist()