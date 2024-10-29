import numpy as np


def robust_mean(data, sigma=2):
    """
    Calculate the mean of the data while excluding outliers that are more than
    a specified number of standard deviations away from the mean.

    Parameters:
    ----------
    data : array-like
        Input data array containing numerical values.
    sigma : float, optional (default=2)
        The number of standard deviations to use as the cutoff for excluding outliers.
        Data points with absolute deviations from the mean greater than `sigma * std`
        will be excluded from the mean calculation.

    Returns:
    -------
    float
        The robust mean calculated after excluding outliers. If no outliers are
        detected, returns the original mean.

    Notes:
    -----
    - This function assumes that the data follows a roughly normal distribution.
    - If multiple outliers exist, they will all be excluded based on the `sigma` threshold.
    - The function performs a single iteration of outlier exclusion. For datasets with
      multiple extreme outliers, consider implementing iterative outlier removal.
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # Identify outliers
    mask = np.abs(data - mean) <= sigma * std
    
    # Exclude outliers and recompute mean
    if np.any(~mask):
        robust_mean = np.mean(data[mask])
        return robust_mean
    else:
        return mean

import numpy as np

def double_robust_mean(data, sigma=2):
    """
    Calculate the mean of the data while excluding outliers more than a specified number
    of standard deviations away from the mean. The outlier exclusion process is performed twice.

    Parameters:
    ----------
    data : array-like
        Input data array containing numerical values.
    sigma : float, optional (default=2)
        The number of standard deviations to use as the cutoff for excluding outliers.
        Data points with absolute deviations from the mean greater than `sigma * std`
        will be excluded from the mean calculation in each iteration.

    Returns:
    -------
    float
        The robust mean calculated after excluding outliers twice. If no outliers are
        detected in either iteration, returns the original mean.

    Notes:
    -----
    - This function assumes that the data follows a roughly normal distribution.
    - Outliers are excluded based on the `sigma` threshold in each iteration.
    - The function performs two iterations of outlier exclusion for enhanced robustness.
    """
    data = np.asarray(data)
    
    # First iteration
    mean1 = np.mean(data)
    std1 = np.std(data)
    mask1 = np.abs(data - mean1) <= sigma * std1
    data1 = data[mask1]

    if not np.any(mask1):
        # All data points are outliers in the first iteration
        return mean1

    # Second iteration on the filtered data
    mean2 = np.mean(data1)
    std2 = np.std(data1)
    mask2 = np.abs(data1 - mean2) <= sigma * std2
    data2 = data1[mask2]

    if not np.any(mask2):
        # All data points are outliers in the second iteration
        return mean2

    # Final robust mean after two iterations
    robust_mean = np.mean(data2)
    return robust_mean

def robust_std(data, sigma=2):
    """
    Calculate the standard deviation of the data while excluding outliers that are more than
    a specified number of standard deviations away from the mean.

    Parameters:
    ----------
    data : array-like
        Input data array containing numerical values.
    sigma : float, optional (default=2)
        The number of standard deviations to use as the cutoff for excluding outliers.
        Data points with absolute deviations from the mean greater than `sigma * std`
        will be excluded from the standard deviation calculation.

    Returns:
    -------
    float
        The robust standard deviation calculated after excluding outliers. If no outliers are
        detected, returns the original standard deviation.

    Notes:
    -----
    - This function assumes that the data follows a roughly normal distribution.
    - If multiple outliers exist, they will all be excluded based on the `sigma` threshold.
    - The function performs a single iteration of outlier exclusion. For datasets with
      multiple extreme outliers, consider implementing iterative outlier removal.
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # Identify outliers
    mask = np.abs(data - mean) <= sigma * std
    
    # Exclude outliers and recompute standard deviation
    if np.any(~mask):
        robust_standard_deviation = np.std(data[mask])
        return robust_standard_deviation
    else:
        return std
