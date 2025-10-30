# 🚀 Quick Start Guide: Using Kaggle Datasets with Your MI Modeling System

## 📥 **Step 1: Install Dependencies**

```bash
# Install Kaggle API
pip install kaggle pandas numpy matplotlib

# Make scripts executable (already done)
chmod +x scripts/*.py
```

## 🔑 **Step 2: Setup Kaggle API**

1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place in `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 📊 **Step 3: Download Datasets**

### Option A: Download MIT-BIH Dataset (Recommended)
```bash
cd /home/ned/projects/cpp/mcmodel
python scripts/download_kaggle_datasets.py --dataset shayanfazeli/heartbeat
```

### Option B: Create Sample Dataset (Quick Test)
```bash
python scripts/download_kaggle_datasets.py --create-sample --samples 1000
```

### Option C: Download PTB-XL Dataset (Large)
```bash
python scripts/download_kaggle_datasets.py --dataset bashayer/ecg-dataset
```

## 🧪 **Step 4: Test Your Dataset**

```bash
# Test the downloaded dataset
python scripts/test_kaggle_data.py datasets/converted_mitbih_train.csv --visualize --test-web

# Or test your sample dataset
python scripts/test_kaggle_data.py datasets/sample_ecg.csv --visualize --test-web
```

## 🌐 **Step 5: Use in WebAssembly Interface**

1. **Start the server** (if not already running):
   ```bash
   cd webassembly && python3 -m http.server 8080
   ```

2. **Open your browser**: http://localhost:8080

3. **Upload your dataset**: Use the file from `datasets/` directory

4. **Run simulations**: Test with different parameters

## 📋 **Recommended Kaggle Datasets**

| Dataset | Size | Best For | Download Command |
|---------|------|----------|------------------|
| **MIT-BIH Arrhythmia** | 40MB | Basic arrhythmia detection | `--dataset shayanfazeli/heartbeat` |
| **ECG Heartbeat Categorization** | 100MB | Multi-class classification | `--dataset shayanfazeli/heartbeat` |
| **PTB-XL ECG Database** | 5GB | Comprehensive cardiac modeling | `--dataset bashayer/ecg-dataset` |
| **Heart Failure Prediction** | 10KB | Clinical validation | `--dataset fedesoriano/heart-failure-prediction` |

## 🎯 **Expected Results**

### MIT-BIH Dataset:
- **Normal beats**: Should show stable patterns
- **MI patterns**: Should show elevated ST segments
- **Arrhythmias**: Should show irregular rhythms

### Performance Targets:
- **Accuracy**: 90%+ for normal vs abnormal
- **Sensitivity**: 85%+ for MI detection
- **Specificity**: 90%+ for normal detection

## 🔧 **Troubleshooting**

### Common Issues:

1. **"Kaggle API not found"**
   ```bash
   pip install kaggle
   ```

2. **"Authentication failed"**
   - Check `~/.kaggle/kaggle.json` exists
   - Verify file permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **"Dataset not found"**
   - Check dataset name is correct
   - Ensure dataset is public

4. **"CSV format error"**
   - Use the conversion script to fix format
   - Check for missing columns

## 📊 **Validation Checklist**

- [ ] Dataset downloaded successfully
- [ ] CSV format validated
- [ ] 12-lead ECG data present
- [ ] Time column exists
- [ ] No NaN values
- [ ] Sample rate consistent (360 Hz)
- [ ] Visualization created
- [ ] WebAssembly test passed

## 🚀 **Advanced Usage**

### Custom Dataset Creation:
```python
from scripts.download_kaggle_datasets import create_sample_dataset
create_sample_dataset("my_custom_dataset.csv", n_samples=5000)
```

### Batch Processing:
```bash
# Process multiple datasets
for dataset in datasets/*.csv; do
    python scripts/test_kaggle_data.py "$dataset" --visualize
done
```

### Integration with Your Model:
```python
import pandas as pd
from your_mi_model import MIModel

# Load dataset
df = pd.read_csv("datasets/converted_mitbih_train.csv")

# Initialize your model
model = MIModel()

# Run analysis
results = model.analyze_ecg(df)
print(f"MI Probability: {results['mi_probability']:.2f}")
```

## 📚 **Additional Resources**

- **Full Guide**: `docs/KAGGLE_DATASET_GUIDE.md`
- **Script Documentation**: `scripts/README.md`
- **WebAssembly Interface**: http://localhost:8080
- **Test Page**: http://localhost:8080/test_metrics.html

## 🎉 **Success Indicators**

You'll know everything is working when:
- ✅ Datasets download without errors
- ✅ CSV files are in correct format
- ✅ Visualizations show realistic ECG patterns
- ✅ WebAssembly interface accepts the data
- ✅ Simulations run without errors
- ✅ Classification metrics display properly

**Happy testing! 🏥📊**




