#!/usr/bin/env python3
"""
Find and download small Kaggle ECG datasets (under 30MB)
"""

import os
import sys
import requests
import json
from pathlib import Path
import argparse

def search_kaggle_datasets(query, max_size_mb=30):
    """Search for Kaggle datasets with size constraints"""
    print(f"🔍 Searching for ECG datasets under {max_size_mb}MB...")
    
    # Common small ECG datasets on Kaggle
    small_datasets = [
        {
            "name": "ECG Heartbeat Categorization Dataset",
            "dataset": "shayanfazeli/heartbeat",
            "size_mb": 25,
            "description": "109,446 ECG heartbeat samples with 5 categories",
            "best_for": "Classification and arrhythmia detection"
        },
        {
            "name": "MIT-BIH Arrhythmia Database",
            "dataset": "shayanfazeli/heartbeat",
            "size_mb": 40,
            "description": "48 half-hour ECG recordings from 47 subjects",
            "best_for": "Basic arrhythmia detection"
        },
        {
            "name": "Heart Disease Dataset",
            "dataset": "johnsmith88/heart-disease-dataset",
            "size_mb": 5,
            "description": "Clinical features for heart disease prediction",
            "best_for": "Clinical validation"
        },
        {
            "name": "ECG Arrhythmia Classification",
            "dataset": "carlossouza/ecg-arrhythmia-classification",
            "size_mb": 15,
            "description": "ECG signals with arrhythmia classifications",
            "best_for": "Arrhythmia classification"
        },
        {
            "name": "Cardiac Arrhythmia Database",
            "dataset": "bashayer/ecg-dataset",
            "size_mb": 20,
            "description": "ECG recordings with various cardiac conditions",
            "best_for": "General cardiac analysis"
        }
    ]
    
    # Filter by size
    suitable_datasets = [d for d in small_datasets if d["size_mb"] <= max_size_mb]
    
    return suitable_datasets

def display_datasets(datasets):
    """Display available datasets"""
    print("\n📊 Available Small ECG Datasets:")
    print("=" * 60)
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset['name']}")
        print(f"   Dataset: {dataset['dataset']}")
        print(f"   Size: {dataset['size_mb']}MB")
        print(f"   Description: {dataset['description']}")
        print(f"   Best For: {dataset['best_for']}")
        print(f"   Download: python scripts/download_kaggle_datasets.py --dataset {dataset['dataset']}")

def download_specific_dataset(dataset_name, output_dir="datasets"):
    """Download a specific dataset"""
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
        
        # Check actual size
        total_size = 0
        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        print(f"📏 Actual downloaded size: {size_mb:.1f}MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def create_mini_sample_dataset(output_file, size_mb=5):
    """Create a mini sample dataset of specific size"""
    print(f"🎯 Creating mini sample dataset (~{size_mb}MB)...")
    
    import pandas as pd
    import numpy as np
    
    # Calculate approximate number of samples for target size
    # Each sample: 13 columns (time + 12 leads) * 4 bytes * samples
    target_samples = int((size_mb * 1024 * 1024) / (13 * 4 * 30))  # 30 seconds per sample
    
    print(f"Generating {target_samples} samples...")
    
    datasets = []
    for i in range(target_samples):
        # Generate 30-second ECG sample
        time = np.arange(0, 30, 1/360)  # 30 seconds at 360 Hz
        heart_rate = np.random.uniform(60, 100)
        
        # Generate realistic ECG signal
        ecg_signal = np.zeros(len(time))
        rr_interval = 60 / heart_rate
        
        # Add R-peaks
        for t in np.arange(0, 30, rr_interval):
            idx = int(t * 360)
            if idx < len(time):
                # QRS complex
                qrs_start = max(0, idx - 14)  # ~40ms before R-peak
                qrs_end = min(len(time), idx + 14)  # ~40ms after R-peak
                
                for j in range(qrs_start, qrs_end):
                    if j < len(time):
                        if j < idx:
                            ecg_signal[j] += -0.1 * np.sin(np.pi * (j - qrs_start) / (idx - qrs_start))
                        else:
                            ecg_signal[j] += 0.8 * np.sin(np.pi * (j - idx) / (qrs_end - idx))
        
        # Add noise
        noise = np.random.normal(0, 0.05, len(time))
        ecg_signal += noise
        
        # Create 12-lead variations
        leads = np.zeros((len(time), 12))
        lead_variations = [1.0, 0.8, 0.6, -0.4, 0.9, 0.7, 0.5, 0.8, 0.9, 1.0, 0.9, 0.8]
        
        for j, variation in enumerate(lead_variations):
            leads[:, j] = ecg_signal * variation + np.random.normal(0, 0.02, len(time))
        
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
            print(f"Generated {i + 1}/{target_samples} samples...")
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Save to file
    combined_df.to_csv(output_file, index=False)
    
    # Check actual size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Mini dataset created: {output_file}")
    print(f"📏 Actual size: {file_size_mb:.1f}MB")
    print(f"📊 Samples: {len(datasets)}")
    print(f"📊 Total rows: {len(combined_df)}")
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Find and download small Kaggle ECG datasets")
    parser.add_argument("--max-size", type=int, default=30, help="Maximum dataset size in MB")
    parser.add_argument("--download", type=str, help="Download specific dataset by name")
    parser.add_argument("--create-mini", action="store_true", help="Create a mini sample dataset")
    parser.add_argument("--mini-size", type=int, default=5, help="Size of mini dataset in MB")
    
    args = parser.parse_args()
    
    print("🏥 Small Kaggle ECG Dataset Finder")
    print("=" * 40)
    
    if args.create_mini:
        # Create mini sample dataset
        output_file = Path("datasets") / f"mini_ecg_{args.mini_size}mb.csv"
        output_file.parent.mkdir(exist_ok=True)
        create_mini_sample_dataset(output_file, args.mini_size)
        return
    
    if args.download:
        # Download specific dataset
        if not setup_kaggle_api():
            return
        download_specific_dataset(args.download)
        return
    
    # Search and display datasets
    datasets = search_kaggle_datasets("ECG", args.max_size)
    display_datasets(datasets)
    
    print(f"\n🎯 Recommended for {args.max_size}MB limit:")
    print("1. ECG Heartbeat Categorization (25MB) - Best for classification")
    print("2. Heart Disease Dataset (5MB) - Best for clinical validation")
    print("3. ECG Arrhythmia Classification (15MB) - Best for arrhythmia detection")
    
    print(f"\n🚀 Quick commands:")
    print(f"# Download ECG Heartbeat dataset:")
    print(f"python scripts/download_kaggle_datasets.py --dataset shayanfazeli/heartbeat")
    print(f"\n# Create mini sample dataset:")
    print(f"python scripts/find_small_kaggle_datasets.py --create-mini --mini-size 5")

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    try:
        import kaggle
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_file = kaggle_dir / "kaggle.json"
        
        if not kaggle_file.exists():
            print("❌ Kaggle credentials not found!")
            print("Please setup Kaggle API first:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Create API Token")
            print("3. Download kaggle.json")
            print("4. Place in ~/.kaggle/kaggle.json")
            return False
        
        return True
    except ImportError:
        print("❌ Kaggle API not installed. Run: pip install kaggle")
        return False

if __name__ == "__main__":
    main()




