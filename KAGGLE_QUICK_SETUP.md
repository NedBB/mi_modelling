# 🚀 Quick Kaggle API Setup

## Step 1: Get Your API Token
1. Go to: https://www.kaggle.com/account
2. Sign in (create account if needed)
3. Scroll to "API" section
4. Click "Create New API Token"
5. Download `kaggle.json`

## Step 2: Setup Token
```bash
# Copy the downloaded file
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

## Step 3: Test Setup
```bash
kaggle datasets list --search "ECG" --size 10
```

## Step 4: Download Datasets
```bash
# ECG Heartbeat (25MB)
python3 scripts/download_kaggle_datasets.py --dataset shayanfazeli/heartbeat

# ECG Arrhythmia (15MB)
python3 scripts/download_kaggle_datasets.py --dataset carlossouza/ecg-arrhythmia-classification

# Heart Disease (5MB)
python3 scripts/download_kaggle_datasets.py --dataset johnsmith88/heart-disease-dataset
```

## Alternative: Create More Sample Datasets
If you want to test without Kaggle setup:
```bash
# Create different sample sizes
python3 scripts/create_simple_sample.py  # Already done (948KB)
python3 scripts/find_small_kaggle_datasets.py --create-mini --mini-size 10  # 10MB
python3 scripts/find_small_kaggle_datasets.py --create-mini --mini-size 20  # 20MB
```




