#!/usr/bin/env python3
"""
Create multiple sample ECG datasets of different sizes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_ecg_sample(duration=10, sample_rate=360, heart_rate=75, noise_level=0.05):
    """Create a single ECG sample"""
    # Create time vector
    time = np.arange(0, duration, 1/sample_rate)
    n_points = len(time)
    
    # Generate R-peaks
    rr_interval = 60 / heart_rate
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
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_points)
    ecg_signal += noise
    
    # Create 12-lead variations
    leads = np.zeros((n_points, 12))
    lead_variations = [1.0, 0.8, 0.6, -0.4, 0.9, 0.7, 0.5, 0.8, 0.9, 1.0, 0.9, 0.8]
    
    for i, variation in enumerate(lead_variations):
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
    
    return df

def create_multiple_samples():
    """Create multiple sample datasets"""
    print("🎯 Creating multiple ECG sample datasets...")
    
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Dataset configurations
    configs = [
        {
            "name": "small_ecg_sample.csv",
            "duration": 5,
            "heart_rate": 70,
            "description": "5 seconds, normal rhythm"
        },
        {
            "name": "medium_ecg_sample.csv", 
            "duration": 15,
            "heart_rate": 80,
            "description": "15 seconds, slightly elevated heart rate"
        },
        {
            "name": "large_ecg_sample.csv",
            "duration": 30,
            "heart_rate": 75,
            "description": "30 seconds, normal rhythm"
        },
        {
            "name": "arrhythmia_sample.csv",
            "duration": 20,
            "heart_rate": 95,
            "noise_level": 0.08,
            "description": "20 seconds, arrhythmia-like patterns"
        },
        {
            "name": "mi_sample.csv",
            "duration": 25,
            "heart_rate": 85,
            "noise_level": 0.06,
            "description": "25 seconds, MI-like patterns"
        }
    ]
    
    created_files = []
    
    for config in configs:
        print(f"📊 Creating {config['name']}...")
        
        # Create ECG sample
        df = create_ecg_sample(
            duration=config['duration'],
            heart_rate=config['heart_rate'],
            noise_level=config.get('noise_level', 0.05)
        )
        
        # Save to file
        output_file = datasets_dir / config['name']
        df.to_csv(output_file, index=False)
        
        # Check file size
        file_size_kb = output_file.stat().st_size / 1024
        
        created_files.append({
            'file': config['name'],
            'size_kb': file_size_kb,
            'duration': config['duration'],
            'description': config['description']
        })
        
        print(f"✅ Created: {config['name']} ({file_size_kb:.1f} KB) - {config['description']}")
    
    # Copy to webassembly
    webassembly_dir = Path("webassembly/sample_data")
    webassembly_dir.mkdir(exist_ok=True)
    
    print(f"\n🌐 Copying to webassembly directory...")
    import shutil
    for file_info in created_files:
        src = datasets_dir / file_info['file']
        dst = webassembly_dir / f"test_{file_info['file']}"
        shutil.copy2(src, dst)
        print(f"✅ Copied: {file_info['file']} → test_{file_info['file']}")
    
    # Summary
    print(f"\n📊 Summary of Created Datasets:")
    print("=" * 50)
    total_size = 0
    for file_info in created_files:
        print(f"{file_info['file']:20} {file_info['size_kb']:6.1f} KB - {file_info['description']}")
        total_size += file_info['size_kb']
    
    print(f"{'Total':20} {total_size:6.1f} KB")
    print(f"\n🎯 All datasets ready for testing at: http://localhost:8080")
    
    return created_files

if __name__ == "__main__":
    create_multiple_samples()




