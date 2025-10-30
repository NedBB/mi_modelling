# 🔑 Kaggle API Setup Guide

## Step 1: Get Your Kaggle API Token

1. **Go to Kaggle**: https://www.kaggle.com/account
2. **Sign in** to your Kaggle account (create one if needed)
3. **Scroll down** to the "API" section
4. **Click "Create New API Token"**
5. **Download** the `kaggle.json` file

## Step 2: Setup the Token

```bash
# Create Kaggle directory (already done)
mkdir -p ~/.kaggle

# Copy your downloaded kaggle.json to the directory
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

## Step 3: Test the Setup

```bash
# Test Kaggle API
kaggle datasets list --search "ECG" --size 10
```

## Step 4: Download ECG Heartbeat Dataset

```bash
# Download the 25MB ECG Heartbeat dataset
python3 scripts/download_kaggle_datasets.py --dataset shayanfazeli/heartbeat
```

## Alternative: Use the Sample Dataset

If you want to test immediately without Kaggle setup:

```bash
# Use the simple sample we just created
python3 scripts/test_kaggle_data.py datasets/simple_ecg_sample.csv --test-web
```

## Quick Test Commands

```bash
# Test the sample dataset
python3 scripts/test_kaggle_data.py datasets/simple_ecg_sample.csv --visualize

# Copy to webassembly for testing
cp datasets/simple_ecg_sample.csv webassembly/sample_data/

# Test in browser
# Go to: http://localhost:8080
# Upload: simple_ecg_sample.csv
```

## Expected Results

- **Sample Dataset**: 944 KB, 10 seconds of ECG data
- **Kaggle Dataset**: 25 MB, 109,446 ECG heartbeat samples
- **Format**: 12-lead ECG in your system's expected format
- **Ready for**: Upload to your webassembly interface




