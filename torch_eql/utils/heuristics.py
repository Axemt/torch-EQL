import pandas as pd
from scipy.signal import correlate
import numpy as np
from math import ceil
from typing import List, Literal

def autocorrelation(grads: pd.Series):
    """
    Compute the autocorrelation of a time series data to detect periodicity.
    """
    correlation = correlate(grads, grads, mode='same')
    mid = len(correlation) // 2
    correlation = correlation[mid:]  # Keep only the positive lag
    correlation = correlation / np.max(correlation)  # Normalize
    return correlation

def detect_periodicity(
        grads: pd.Series,
        method: Literal['autocorrelation', 'value'] = 'autocorrelation',
        default_window_size: int = 20
    ) -> int:
    """
    Detect the periodicity (window size) of a time series using autocorrelation or from gradient values.
    """

    peak_source = None
    if method == 'autocorrelation':
        peak_source = autocorrelation(grads)
    elif method == 'value':
        peak_source = grads
    else:
        raise ValueError(
            f'Periodicity method `{method}` is not supported'
        )
    
    # To detect peaks, we compare each element to its neighbors
    peaks = []
    for i in range(1, len(peak_source) - 1):
        if peak_source[i] > peak_source[i - 1] and peak_source[i] > peak_source[i + 1]:
            peaks.append(i)

    # If no peaks are found, return a default window size (e.g., 20)
    if len(peaks) > 1:
        peak_distances = pd.Series(peaks) - pd.Series(peaks).shift(1, fill_value=peaks[0])
        window_size = int(peak_distances.max())
    else:
        window_size = default_window_size

    return window_size

def all_signal_tendencies(metric: pd.Series, band_width: int):

    return np.array([
        signal_tendency_at(metric, i, band_width)
        for i in range(len(metric))
    ])

def signal_tendency_at(metric: pd.Series, at: int, band_width: int):

    if at <= band_width: return float('nan')
    return sum(metric.iloc[at] > metric.iloc[at-band_width: at]) / band_width
    
def get_moving(
        s: pd.Series, 
        period: int, 
        method: Literal['mean', 'median'] = 'median'
    ) -> pd.Series:

    m = s.rolling(period)

    if method == 'mean':
        return m.mean()
    elif method == 'median':
        return m.median()
    else:
        raise ValueError(
            f'Moving method `{method}` is not a valid argument'
        )

def signal_from_tracked(
        tracked: List[float] | List[List[float]],
        total_epochs: int,
        period: int | None = None,
        period_width: int | None = None,
        method: Literal['mean', 'median'] = 'median'
    ) -> float:

    if isinstance(tracked[0], float):
        tracked = [tracked]
    
    period_width = (2*ceil(np.log10(total_epochs))) if period_width is None else period_width

    agg_signal = []
    for g in tracked:
        if g == []: continue
        
        s = pd.Series(np.log10(g))

        period = detect_periodicity(s) if period is None else period
        wide_period = period*period_width

        if len(s) >= wide_period:
            moving = get_moving(s, period, method=method)
            signal = signal_tendency_at(moving, len(moving)-1, wide_period)
        else:
            signal = float('nan')

        agg_signal.append(signal)


    agg_signal = np.mean(agg_signal)
    return agg_signal
