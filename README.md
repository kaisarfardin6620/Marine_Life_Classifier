# **Marine Life Classification with Deep Learning**

This repository contains Python scripts for classifying marine life images using various deep learning approaches, primarily leveraging transfer learning with pre-trained convolutional neural networks (CNNs). The project explores different preprocessing techniques and training strategies to achieve robust classification performance.

## **Table of Contents**

1. Project Description
2. Features
3. Files in this Repository
4. Dataset
5. Installation
6. Usage
7. Expected Outputs
8. Areas for Improvement / Future Work
9. License
10. Acknowledgements
## **Project Description**

This project aims to classify various species of marine life from images using deep learning. It demonstrates three distinct approaches to building and training image classification models:

* A model incorporating image denoising as a preprocessing step.  
* A baseline model using a standard ResNet-101 architecture.  
* A more advanced model employing a two-stage training strategy (feature extraction followed by fine-tuning) with MobileNetV2.

All models utilize transfer learning from pre-trained CNNs, data augmentation, and comprehensive evaluation metrics to ensure robust performance.

## **Features**

* **Transfer Learning:** Utilizes pre-trained ResNet101 and MobileNetV2 models as feature extractors.  
* **Data Augmentation:** Employs ImageDataGenerator for on-the-fly image transformations to enhance model generalization.  
* **Custom Preprocessing:** Includes an option for integrating image denoising (Bilateral Filter) into the preprocessing pipeline.  
* **Two-Stage Training:** Implements a sophisticated training strategy for MobileNetV2, involving initial feature extraction and subsequent fine-tuning of the base model.  
* **Learning Rate Scheduling:** Uses CosineDecay for optimized learning rate management during training.  
* **Callbacks:** Integrates EarlyStopping and ModelCheckpoint to prevent overfitting and save the best model weights.  
* **Comprehensive Evaluation:** Generates detailed classification reports, confusion matrices, and per-class accuracy metrics.  
* **Extensive Visualization:** Provides plots for class distributions, sample augmented images, training history (accuracy, loss), and generalization gap.  
* **Structured Logging:** Uses Python's logging module for informative output during execution.

## **Files in this Repository**

* marine\_denoising.py:  
  * **Purpose:** Builds a marine life classification model using ResNet101 and incorporates **image denoising** (specifically, a Bilateral Filter by default) as a preprocessing step before feeding images to the CNN.  
  * **Key Aspect:** Focuses on improving image quality before classification, which can be beneficial for noisy datasets.  
* marine\_life.py:  
  * **Purpose:** Implements a baseline marine life classification model using ResNet101 without any explicit custom denoising preprocessing.  
  * **Key Aspect:** Serves as a direct comparison to the denoising approach, relying solely on the pre-trained model's ability to handle image characteristics.  
* marine\_life\_2stg.py:  
  * **Purpose:** Develops a marine life classification model using MobileNetV2 with a **two-stage training strategy**: first, training a new classification head with the base model frozen (feature extraction), and then unfreezing and fine-tuning parts of the base model along with the head.  
  * **Key Aspect:** Demonstrates a more advanced and often more effective transfer learning technique for optimizing model performance.

## **Dataset**

This project expects the dataset to be organized in a specific folder structure, typically found in image classification tasks. The base\_path variable (set to 'marine\_life\_split' in the scripts) should point to the root directory of your dataset.

The expected structure is:

marine\_life\_split/  
├── train/  
│   ├── class\_name\_1/  
│   │   ├── image1.jpg  
│   │   ├── image2.png  
│   │   └── ...  
│   ├── class\_name\_2/  
│   │   └── ...  
│   └── ...  
├── test/  
│   ├── class\_name\_1/  
│   │   └── ...  
│   └── ...  
└── val/  
    ├── class\_name\_1/  
    │   └── ...  
    └── ...

Each class\_name\_X folder should contain images belonging to that specific class.

## **Installation**

1. **Clone the repository:**  
   git clone https://github.com/kaisarfardin6620/Marine_Life_Classifier.git  
   cd your-repo-name

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows: \`venv\\Scripts\\activate\`

3. **Install the required libraries:**  
   pip install tensorflow keras pandas numpy matplotlib seaborn scikit-image opencv-python pillow scikit-learn

   * **Note:** TensorFlow and Keras versions should be compatible. The scripts were developed with tensorflow==2.x and keras (part of TensorFlow).

## **Usage**

Before running any script, ensure your dataset is organized as described in the [Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste?select=Clams) section and placed in the marine\_life\_split directory (or update the base\_path variable in each script accordingly).

To run any of the models, simply execute the corresponding Python script:

* **To run the denoising model:**  
  python marine\_denoising.py

* **To run the baseline ResNet-101 model:**  
  python marine\_life.py

* **To run the two-stage MobileNetV2 model:**  
  python marine\_life\_2stg.py

Each script will print training progress, evaluation results, and display various plots.

## **Expected Outputs**

Upon running the scripts, you can expect the following:

* **Console Output:**  
  * Logging messages detailing data loading, model summary, training progress (epoch-wise loss and accuracy), and final evaluation metrics.  
  * Classification reports showing precision, recall, and F1-score for each class.  
* **Plots (displayed and can be saved manually):**  
  * Overall and split-wise class distributions.  
  * Sample augmented images from the training set.  
  * Training and validation accuracy/loss curves over epochs.  
  * Confusion matrices visualizing model performance across classes.  
  * Per-class accuracy bar plots (in marine\_life\_2stg.py).  
  * Learning rate schedule plot (in marine\_life\_2stg.py).  
* **Saved Models:**  
  * marine\_life\_resnet101.keras (from marine\_denoising.py and marine\_life.py): The best performing model during training.  
  * best\_marine\_life\_model\_feature\_extraction.keras (from marine\_life\_2stg.py): Best model after the feature extraction phase.  
  * best\_marine\_life\_finetuned.keras (from marine\_life\_2stg.py): Best model after the fine-tuning phase.  
  * final\_marine\_life\_model.keras (from marine\_life\_2stg.py): The final trained model.

## **Areas for Improvement / Future Work**

* **Hyperparameter Tuning:** Systematically tune learning rates, dropout rates, regularization strengths, and data augmentation parameters for optimal performance.  
* **Denoising Configuration:** In marine\_denoising.py, make the choice of denoising filter (Bilateral, Median, Gaussian, Non-Local Means) configurable via arguments or a clear flag.  
* **Dynamic NUM\_CLASSES:** Instead of hardcoding NUM\_CLASSES, derive it dynamically from the ImageDataGenerator's num\_classes attribute to ensure robustness against dataset changes.  
* **Advanced Data Augmentation:** Explore more advanced augmentation techniques or libraries (e.g., Albumentations) for potentially better results.  
* **Model Architectures:** Experiment with other state-of-the-art CNN architectures (e.g., EfficientNet, Vision Transformers).  
* **Cross-Validation:** Implement k-fold cross-validation for more reliable performance estimation.  
* **Deployment:** Consider deploying the trained model using frameworks like TensorFlow Serving or Flask/FastAPI for inference.  
* **Code Refactoring:** Consolidate common functions and configurations into shared utility files to reduce redundancy across the scripts.


## **Acknowledgements**

* This project utilizes pre-trained models from the TensorFlow Keras Applications library.  
* Data preprocessing and visualization leverage popular Python libraries like OpenCV, scikit-image, Pandas, NumPy, Matplotlib, and Seaborn.
