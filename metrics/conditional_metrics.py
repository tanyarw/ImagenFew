import numpy as np
import torch
from metrics.discriminative_torch import discriminative_score_metrics
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def get_active_mask(data, threshold, channel_idx=-1):
    """Returns a mask of sequences that contain at least one value above threshold."""
    # data: [batch, seq_len, channels]
    return np.any(data[:, :, channel_idx] > threshold, axis=1)

def calculate_conditional_metrics(real_data, gen_data, device, threshold=0.1, channel_idx=-1):
    """
    Computes metrics focusing on non-zero/active events.
    """
    # 1. Wet-Window Discriminative Score
    real_mask = get_active_mask(real_data, threshold, channel_idx)
    gen_mask = get_active_mask(gen_data, threshold, channel_idx)
    
    real_wet = real_data[real_mask]
    gen_wet = gen_data[gen_mask]
    
    results = {}
    
    if len(real_wet) > 10 and len(gen_wet) > 10:
        results['cond_disc_score'] = discriminative_score_metrics(real_wet, gen_wet, device)
    else:
        results['cond_disc_score'] = -1.0 # Not enough samples
        
    # 2. Non-Zero Intensity Distribution (JSD)
    real_vals = real_data[:, :, channel_idx].flatten()
    gen_vals = gen_data[:, :, channel_idx].flatten()
    
    real_nonzero = real_vals[real_vals > threshold]
    gen_nonzero = gen_vals[gen_vals > threshold]
    
    if len(real_nonzero) > 0 and len(gen_nonzero) > 0:
        # Create histograms for JSD
        bins = np.linspace(threshold, max(real_vals.max(), gen_vals.max()), 100)
        p, _ = np.histogram(real_nonzero, bins=bins, density=True)
        q, _ = np.histogram(gen_nonzero, bins=bins, density=True)
        # Add small epsilon to avoid log(0)
        p += 1e-10
        q += 1e-10
        results['intensity_jsd'] = jensenshannon(p, q)
    else:
        results['intensity_jsd'] = -1.0

    # 3. Extreme Value Capture (95th/99th Percentile Ratio)
    if len(gen_nonzero) > 0:
        results['real_p99'] = np.percentile(real_nonzero, 99)
        results['gen_p99'] = np.percentile(gen_nonzero, 99)
        results['extreme_ratio'] = results['gen_p99'] / results['real_p99']
    else:
        results['extreme_ratio'] = 0.0

    # 4. Event Duration Stats
    def get_durations(data_flat, thresh):
        is_wet = (data_flat > thresh).astype(int)
        # Find changes from dry to wet and vice versa
        diff = np.diff(np.concatenate(([0], is_wet, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return ends - starts

    real_durs = get_durations(real_vals, threshold)
    gen_durs = get_durations(gen_vals, threshold)
    
    results['avg_real_duration'] = np.mean(real_durs) if len(real_durs) > 0 else 0
    results['avg_gen_duration'] = np.mean(gen_durs) if len(gen_durs) > 0 else 0
    
    return results
