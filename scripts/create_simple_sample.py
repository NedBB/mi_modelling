#!/usr/bin/env python3
"""
Create a simple sample ECG dataset for testing
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_simple_sample_dataset(output_file="datasets/simple_ecg_sample.csv", duration=10, sample_rate=360):
    """Create a simple ECG sample dataset"""
    print(f"🎯 Creating simple ECG sample dataset...")
    
    # Create time vector
    time = np.arange(0, duration, 1/sample_rate)
    n_points = len(time)
    
    # Generate base ECG signal with realistic heart rate
    heart_rate = 75  # BPM
    rr_interval = 60 / heart_rate
    
    # Generate R-peaks
    r_peaks = np.arange(0, duration, rr_interval)
    
    # Create ECG signal
    ecg_signal = np.zeros(n_points)
    
    for peak_time in r_peaks:
        peak_idx = int(peak_time * sample_rate)
        if peak_idx < n_points:
            # QRS complex
            qrs_start = max(0, peak_idx - int(0.04 * sample_rate))
            qrs_end = min(n_points, peak_idx + int(0.04 * sample_rate))
            
            for j in range(qrs_start, qrs_end):
                if j < n_points:
                    # QRS shape
                    if j < peak_idx:
                        ecg_signal[j] += -0.1 * np.sin(np.pi * (j - qrs_start) / (peak_idx - qrs_start))
                    else:
                        ecg_signal[j] += 0.8 * np.sin(np.pi * (j - peak_idx) / (qrs_end - peak_idx))
    
    # Add some noise
    noise = np.random.normal(0, 0.05, n_points)
    ecg_signal += noise
    
    # Create 12-lead variations (realistic ECG lead relationships)
    leads = np.zeros((n_points, 12))
    lead_variations = [
        1.0,    # Lead I
        0.8,    # Lead II  
        0.6,    # Lead III
        -0.4,   # Lead aVR
        0.9,    # Lead aVL
        0.7,    # Lead aVF
        0.5,    # Lead V1
        0.8,    # Lead V2
        0.9,    # Lead V3
        1.0,    # Lead V4
        0.9,    # Lead V5
        0.8     # Lead V6
    ]
    
    for i, variation in enumerate(lead_variations):
        # Add some noise and variation to make it realistic
        noise = np.random.normal(0, 0.02, n_points)
        leads[:, i] = ecg_signal * variation + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'lead_I': leads[:, 0],
        'lead_II': leads[:, 1],
        'lead_III': leads[:, 2],
        'lead_aVR': leads[:, 3],
        'lead_aVL': leads[:, 4],
        'lead_aVF': leads[:, 5],
        'lead_V1': leads[:, 6],
        'lead_V2': leads[:, 7],
        'lead_V3': leads[:, 8],
        'lead_V4': leads[:, 9],
        'lead_V5': leads[:, 10],
        'lead_V6': leads[:, 11]
    })
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Check file size
    file_size_kb = output_path.stat().st_size / 1024
    
    print(f"✅ Simple ECG sample created: {output_file}")
    print(f"📏 File size: {file_size_kb:.1f} KB")
    print(f"📊 Duration: {duration} seconds")
    print(f"📊 Sample rate: {sample_rate} Hz")
    print(f"📊 Total points: {len(df)}")
    
    return df

if __name__ == "__main__":
    create_simple_sample_dataset()




