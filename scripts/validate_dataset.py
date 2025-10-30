#!/usr/bin/env python3
"""
Simple dataset validation script
"""

import pandas as pd
import numpy as np
from pathlib import Path

def validate_dataset(csv_file):
    """Validate dataset format"""
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
        
        # Dataset statistics
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

def copy_to_webassembly(csv_file):
    """Copy dataset to webassembly directory"""
    print(f"🌐 Copying to webassembly directory...")
    
    webassembly_dir = Path("webassembly/sample_data")
    webassembly_dir.mkdir(exist_ok=True)
    
    import shutil
    test_file = webassembly_dir / f"test_{Path(csv_file).name}"
    shutil.copy2(csv_file, test_file)
    
    print(f"✅ Test file copied to: {test_file}")
    print(f"🌐 You can now test it at: http://localhost:8080")
    print(f"📁 Upload file: {test_file.name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python3 validate_dataset.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    print("🧪 Validating ECG Dataset")
    print("=" * 30)
    
    if validate_dataset(csv_file):
        copy_to_webassembly(csv_file)
        print("\n✅ Dataset is ready for testing!")
    else:
        print("\n❌ Dataset validation failed")




