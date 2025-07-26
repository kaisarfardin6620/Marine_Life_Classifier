# 🧠 CNN Dataset Preprocessing Pipeline

This Jupyter Notebook (`cnn_preprocessings.ipynb`) provides a complete and modular pipeline for preparing image datasets for Convolutional Neural Network (CNN) training. It is ideal for scenarios where you have raw image data organized by class, and need to structure it into training, validation, and test splits before feeding it into a deep learning model.

## 📁 Key Features

### 1. Initial Dataset Overview
- Automatically detects whether your dataset is already split (`train/`, `val/`, `test/`) or still raw (all classes under one directory).
- Displays:
  - Number of classes
  - Number of images per split or per class (if not yet split)

### 2. Dataset Splitting
- Splits your dataset into `train`, `validation`, and `test` folders using a 70/15/15 ratio.
- Maintains class distribution across splits.
- Supports `.jpg`, `.jpeg`, `.png`, and `.bmp` image formats.
- Randomized shuffling ensures good stratification.

### 3. Post-Split Verification
- Counts the number of images in each of the split folders.
- Verifies that splitting was executed correctly.

## ⚙️ Requirements

Make sure you have the following Python packages installed:

```bash
pip install numpy scikit-learn
```

Also, this notebook assumes the following structure for your raw dataset:

```
/your_dataset/
├── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class_2/
│   ├── img1.jpg
│   └── ...
└── ...
```

## 🚀 How to Use

1. Set the paths in the notebook:
   ```python
   base_path = '/path/to/your/raw_dataset'
   output_dir = '/path/to/output_dataset'
   ```

2. Run the notebook step-by-step:
   - Get dataset stats
   - Perform the split
   - Validate the result

3. Use the processed dataset directly with popular libraries like TensorFlow or PyTorch.

## 📌 Notes

- This notebook is not tied to any specific model training; it focuses purely on data preparation.
- You can easily modify the split ratio or supported formats.
- Designed for one-time execution per dataset.

## 📄 License

This project is open-source and available under the MIT License.