#!/usr/bin/env python3
"""
Test script for validating Kaggle datasets with the MI modeling system
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def validate_dataset_format(csv_file):
    """Validate that the dataset is in the correct format"""
    print(f"🔍 Validating dataset: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_columns = [
            'time', 'lead_I', 'lead_II', 'lead_III', 'lead_aVR', 
            'lead_aVL', 'lead_aVF', 'lead_V1', 'lead_V2', 
            'lead_V3', 'lead_V4', 'lead_V5', 'lead_V6'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"❌ Column {col} is not numeric")
                return False
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check data ranges
        print(f"📊 Dataset statistics:")
        print(f"   Rows: {len(df)}")
        print(f"   Duration: {df['time'].max() - df['time'].min():.2f} seconds")
        print(f"   Sample rate: {len(df) / (df['time'].max() - df['time'].min()):.1f} Hz")
        
        # Check signal ranges
        for col in ['lead_I', 'lead_II', 'lead_III']:
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"   {col}: {min_val:.3f} to {max_val:.3f}")
        
        print("✅ Dataset format validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return False

def test_with_webassembly_interface(csv_file):
    """Test the dataset with the webassembly interface"""
    print(f"🌐 Testing with webassembly interface...")
    
    # Copy file to webassembly directory
    webassembly_dir = project_root / "webassembly" / "sample_data"
    webassembly_dir.mkdir(exist_ok=True)
    
    import shutil
    test_file = webassembly_dir / f"test_{Path(csv_file).name}"
    shutil.copy2(csv_file, test_file)
    
    print(f"✅ Test file copied to: {test_file}")
    print(f"🌐 You can now test it at: http://localhost:8080")
    print(f"📁 Upload file: {test_file.name}")

def create_visualization(csv_file, output_dir="visualizations"):
    """Create visualizations of the dataset"""
    print(f"📊 Creating visualizations...")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot all 12 leads
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle(f'ECG Data Visualization - {Path(csv_file).name}', fontsize=16)
        
        lead_columns = ['lead_I', 'lead_II', 'lead_III', 'lead_aVR', 'lead_aVL', 'lead_aVF',
                       'lead_V1', 'lead_V2', 'lead_V3', 'lead_V4', 'lead_V5', 'lead_V6']
        
        for i, col in enumerate(lead_columns):
            row = i // 3
            col_idx = i % 3
            
            axes[row, col_idx].plot(df['time'], df[col])
            axes[row, col_idx].set_title(col)
            axes[row, col_idx].set_xlabel('Time (s)')
            axes[row, col_idx].set_ylabel('Amplitude (mV)')
            axes[row, col_idx].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f"{Path(csv_file).stem}_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to: {plot_file}")
        
        # Create summary statistics
        stats_file = output_path / f"{Path(csv_file).stem}_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Dataset Statistics: {Path(csv_file).name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Duration: {df['time'].max() - df['time'].min():.2f} seconds\n")
            f.write(f"Sample rate: {len(df) / (df['time'].max() - df['time'].min()):.1f} Hz\n\n")
            
            f.write("Lead Statistics:\n")
            f.write("-" * 20 + "\n")
            for col in lead_columns:
                f.write(f"{col}:\n")
                f.write(f"  Mean: {df[col].mean():.4f}\n")
                f.write(f"  Std:  {df[col].std():.4f}\n")
                f.write(f"  Min:  {df[col].min():.4f}\n")
                f.write(f"  Max:  {df[col].max():.4f}\n\n")
        
        print(f"✅ Statistics saved to: {stats_file}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Kaggle datasets with MI modeling system")
    parser.add_argument("csv_file", help="Path to CSV file to test")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--test-web", action="store_true", help="Test with webassembly interface")
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv_file)
    
    if not csv_file.exists():
        print(f"❌ File not found: {csv_file}")
        return
    
    print("🧪 Testing Kaggle Dataset with MI Modeling System")
    print("=" * 60)
    
    # Validate dataset format
    if not validate_dataset_format(csv_file):
        print("❌ Dataset validation failed")
        return
    
    # Create visualizations if requested
    if args.visualize:
        create_visualization(csv_file)
    
    # Test with webassembly interface if requested
    if args.test_web:
        test_with_webassembly_interface(csv_file)
    
    print("\n✅ Dataset testing completed successfully!")
    print("\n📋 Next steps:")
    print("1. Upload the dataset to your webassembly interface")
    print("2. Run simulations with different parameters")
    print("3. Compare results with known ground truth")
    print("4. Validate classification metrics")

if __name__ == "__main__":
    main()




