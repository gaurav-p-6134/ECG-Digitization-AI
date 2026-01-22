import numpy as np
from scipy.signal import savgol_filter

def get_baseline_mode_high_res(signal, bins=10000):
    """
    Calculates the signal baseline using a high-resolution histogram.
    
    Args:
        signal (np.array): 1D array of signal voltage values.
        bins (int): Number of bins for the histogram. Higher = more precision.
        
    Returns:
        float: The estimated baseline voltage (mode of the distribution).
    """
    try:
        hist, bin_edges = np.histogram(signal, bins=bins)
        max_idx = np.argmax(hist)
        # Return the center of the bin with the highest count
        return (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0
    except Exception:
        # Fallback to simple median if histogram fails
        return np.median(signal)

def safe_savgol(y, window=7, poly=3):
    """
    Applies Savitzky-Golay smoothing with edge padding handling.
    
    Args:
        y (np.array): Input signal.
        window (int): Filter window length (must be odd).
        poly (int): Polynomial order.
        
    Returns:
        np.array: Smoothed signal.
    """
    try:
        # Reflect padding handles edge artifacts
        pad_len = window // 2 + 5
        y_pad = np.pad(y, (pad_len, pad_len), mode='reflect')
        y_smooth = savgol_filter(y_pad, window, poly)
        return y_smooth[pad_len:-pad_len]
    except Exception:
        return y

def adaptive_clip(signal, lower_p=0.1, upper_p=99.9, margin=0.5):
    """
    Clips outliers based on the signal's own statistical distribution.
    Prevents noise spikes in low-amplitude leads from skewing metrics.
    """
    try:
        p_low = np.percentile(signal, lower_p)
        p_high = np.percentile(signal, upper_p)
        
        clip_min = max(-4.0, p_low - margin)
        clip_max = min(4.0, p_high + margin)
        
        return np.clip(signal, clip_min, clip_max)
    except Exception:
        return np.clip(signal, -4.0, 4.0)

def apply_einthoven_smoothing(series_mv):
    """
    Refines Limb Leads (I, II, III) and Augmented Leads (aVR, aVL, aVF)
    using Einthoven's Law and Zero-Sum constraints.
    
    Physics Constraints:
        1. II = I + III
        2. aVR + aVL + aVF = 0
        
    Args:
        series_mv (np.array): Shape (4, N). Rows mapped to standard 12-lead layout.
                              Row 0: I, aVR, V1, V4
                              Row 1: II, aVL, V2, V5
                              Row 2: III, aVF, V3, V6
                              
    Returns:
        np.array: Physically corrected signal array.
    """
    try:
        N = series_mv.shape[1]
        quarter = N // 4
        
        # --- Chunk 1: Limb Leads (I, II, III) ---
        lead_I = series_mv[0, 0:quarter]
        lead_II = series_mv[1, 0:quarter]
        lead_III = series_mv[2, 0:quarter]
        
        # Enforce II = I + III (50% Model trust, 50% Physics trust)
        calc_II = lead_I + lead_III
        new_II = (lead_II * 0.6) + (calc_II * 0.4) 
        
        # Recalculate I and III based on the cleaner II
        calc_I = new_II - lead_III
        calc_III = new_II - lead_I
        
        series_mv[0, 0:quarter] = (lead_I * 0.5) + (calc_I * 0.5)
        series_mv[1, 0:quarter] = new_II
        series_mv[2, 0:quarter] = (lead_III * 0.5) + (calc_III * 0.5)
        
        # --- Chunk 2: Augmented Leads (aVR, aVL, aVF) ---
        start, end = quarter, quarter * 2
        
        # Enforce Zero Sum Constraint: aVR + aVL + aVF = 0
        sum_aug = series_mv[0, start:end] + series_mv[1, start:end] + series_mv[2, start:end]
        avg_bias = sum_aug / 3.0
        
        # Remove common mode bias
        series_mv[0, start:end] -= avg_bias
        series_mv[1, start:end] -= avg_bias
        series_mv[2, start:end] -= avg_bias
        
        return series_mv
        
    except Exception as e:
        print(f"Warning: Einthoven smoothing failed: {e}")
        return series_mv