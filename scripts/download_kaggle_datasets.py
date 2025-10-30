#!/usr/bin/env python3
"""
Kaggle Dataset Downloader and Converter for MI Modeling System
"""

import os
import sys
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("🔧 Setting up Kaggle API...")
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle API is installed")
    except ImportError:
        print("❌ Kaggle API not found. Installing...")
        os.system("pip install kaggle")
        import kaggle
    
    # Check for credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if not kaggle_file.exists():
        print("❌ Kaggle credentials not found!")
        print("Please:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create API Token")
        print("3. Download kaggle.json")
        print("4. Place in ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle credentials found")
    return True

def download_dataset(dataset_name, output_dir="datasets"):
    """Download dataset from Kaggle"""
    print(f"📥 Downloading dataset: {dataset_name}")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(dataset_name, path=output_path, unzip=True)
        print(f"✅ Dataset downloaded to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def convert_mitbih_to_system_format(input_file, output_file, sample_rate=360):
    """Convert MIT-BIH dataset to system format"""
    print(f"🔄 Converting {input_file} to system format...")
    
    try:
        # Load MIT-BIH data
        df = pd.read_csv(input_file)
        
        # MIT-BIH typically has single lead data
        if df.shape[1] == 2:  # time, signal
            time_col, signal_col = df.columns[0], df.columns[1]
        else:
            signal_col = df.columns[0]
            time_col = None
        
        # Create time vector if not present
        if time_col is None:
            time = np.arange(len(df)) / sample_rate
        else:
            time = df[time_col].values
        
        # Get signal data
        signal = df[signal_col].values
        
        # Create 12-lead ECG by replicating and adding variations
        leads = np.zeros((len(signal), 12))
        
        # Lead variations (based on standard ECG lead relationships)
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
            noise = np.random.normal(0, 0.05, len(signal))
            leads[:, i] = signal * variation + noise
        
        # Create system format DataFrame
        system_df = pd.DataFrame({
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
        
        # Save converted data
        system_df.to_csv(output_file, index=False)
        print(f"✅ Converted data saved to {output_file}")
        
        return system_df
        
    except Exception as e:
        print(f"❌ Error converting data: {e}")
        return None

def create_sample_dataset(output_file, n_samples=1000, duration=30, sample_rate=360):
    """Create a sample dataset for testing"""
    print(f"🎯 Creating sample dataset with {n_samples} samples...")
    
    # Generate synthetic ECG data
    time = np.arange(0, duration, 1/sample_rate)
    n_points = len(time)
    
    datasets = []
    
    for i in range(n_samples):
        # Generate base ECG signal
        heart_rate = np.random.uniform(60, 100)  # BPM
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
        
        # Create 12-lead variations
        leads = np.zeros((n_points, 12))
        lead_variations = [1.0, 0.8, 0.6, -0.4, 0.9, 0.7, 0.5, 0.8, 0.9, 1.0, 0.9, 0.8]
        
        for j, variation in enumerate(lead_variations):
            leads[:, j] = ecg_signal * variation + np.random.normal(0, 0.02, n_points)
        
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
        
        datasets.append(df)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_samples} samples...")
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Save to file
    combined_df.to_csv(output_file, index=False)
    print(f"✅ Sample dataset saved to {output_file}")
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Download and convert Kaggle datasets for MI modeling")
    parser.add_argument("--dataset", type=str, help="Kaggle dataset name (e.g., 'shayanfazeli/heartbeat')")
    parser.add_argument("--output", type=str, default="datasets", help="Output directory")
    parser.add_argument("--create-sample", action="store_true", help="Create a sample dataset for testing")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples for synthetic dataset")
    
    args = parser.parse_args()
    
    print("🏥 Kaggle Dataset Downloader for MI Modeling System")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.create_sample:
        # Create sample dataset
        output_file = output_dir / "sample_ecg.zip"
        create_sample_dataset(output_file, args.samples)
        return
    
    if args.dataset:
        # Download specific dataset
        if not setup_kaggle_api():
            return
        
        dataset_path = download_dataset(args.dataset, args.output)
        if dataset_path:
            # Find CSV files in the downloaded dataset
            csv_files = list(dataset_path.glob("*.csv"))
            if csv_files:
                # Convert the first CSV file
                input_file = csv_files[0]
                output_file = output_dir / f"converted_{input_file.name}"
                convert_mitbih_to_system_format(input_file, output_file)
            else:
                print("❌ No CSV files found in downloaded dataset")
    else:
        print("📋 Available options:")
        print("1. Download MIT-BIH dataset:")
        print("   python download_kaggle_datasets.py --dataset shayanfazeli/heartbeat")
        print()
        print("2. Create sample dataset:")
        print("   python download_kaggle_datasets.py --create-sample --samples 1000")
        print()
        print("3. Download PTB-XL dataset:")
        print("   python download_kaggle_datasets.py --dataset bashayer/ecg-dataset")
        print()
        print("📚 See docs/KAGGLE_DATASET_GUIDE.md for more information")

if __name__ == "__main__":
    main()




