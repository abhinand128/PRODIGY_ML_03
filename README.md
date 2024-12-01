# **Cat vs Dog Image Classification using SVM**

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Requirements](#requirements)
   - [Libraries](#libraries)
   - [Installation](#installation)
5. [Usage](#usage)
   - [Prepare the Dataset](#1-prepare-the-dataset)
   - [Run the Script](#2-run-the-script)
   - [View Results](#3-view-results)
6. [Code Explanation](#code-explanation)
   - [1. Image Preprocessing](#1-image-preprocessing)
   - [2. Dataset Splitting](#2-dataset-splitting)
   - [3. Model Training](#3-model-training)
   - [4. Prediction and Evaluation](#4-prediction-and-evaluation)
   - [5. Saving Results](#5-saving-results)
7. [Output Example](#output-example)
8. [Limitations](#limitations)

---

## **Overview**
This project is a machine learning pipeline that classifies images of cats and dogs using **Support Vector Machines (SVM)**. It preprocesses the input dataset by converting images to grayscale, resizing them, and flattening them for compatibility with the SVM model. After training, the model outputs predictions and calculates accuracy.

---

## **Features**
- **Data Preprocessing**:
  - Converts images to grayscale and resizes them to \(64 \times 64\) pixels.
  - Flattens the image data for SVM compatibility.
- **Model Training**:
  - Utilizes **Support Vector Machine (SVM)** with a linear kernel for binary classification.
- **Evaluation**:
  - Generates a detailed accuracy score.
  - Saves a CSV file with predictions and comparisons against actual labels.

---

## **Directory Structure**
```plaintext
project/
├── Data_set/
│   └── train/          # Folder containing the training images
├── Predictions.csv     # CSV file with predictions (generated post execution)
├── main.py             # Script with the classification pipeline
└── README.md           # Project documentation
```

---

## **Requirements**

### **Libraries**
The following Python libraries are required:
- **NumPy**: Array operations and data manipulation.
- **pandas**: DataFrame creation and CSV handling.
- **scikit-learn**: Machine learning model and utilities.
- **Pillow (PIL)**: Image loading and manipulation.

### **Installation**
Install the libraries using the following command:
```bash
pip install numpy pandas scikit-learn pillow
```

---

## **Usage**

### **1. Prepare the Dataset**
- Place training images in the `Data_set/train` folder.
- Ensure filenames include keywords like `cat` or `dog` for label inference.
  
### **2. Run the Script**
- Execute the main script:
  ```bash
  python main.py
  ```

### **3. View Results**
- The script will:
  1. Print the **model accuracy**.
  2. Generate a `Predictions.csv` file containing:
     - `Image ID`: Filename of the test image.
     - `Actual Label`: Ground truth label.
     - `Predicted Label`: Model's prediction.

---

## **Code Explanation**

### **1. Image Preprocessing**
- Converts images to **grayscale** for simplicity.
- Resizes images to \(64 \times 64\) pixels for uniformity.
- Flattens the \(64 \times 64\) matrix into a single-dimensional array.

### **2. Dataset Splitting**
- Splits the dataset into:
  - **Training Set**: 80% of the data for model training.
  - **Testing Set**: 20% for evaluating the model's performance.

### **3. Model Training**
- Uses a **Support Vector Machine (SVM)** with a **linear kernel** for binary classification:
  - Label `0`: Represents cats.
  - Label `1`: Represents dogs.

### **4. Prediction and Evaluation**
- Generates predictions for the test set.
- Maps numeric predictions (`0` or `1`) back to string labels (`cat` or `dog`).
- Computes and displays the **accuracy score**.

### **5. Saving Results**
- Outputs predictions to a CSV file (`Predictions.csv`) with the following columns:
  - `Image ID`: Test image filename.
  - `Actual Label`: Ground truth label.
  - `Predicted Label`: Model's prediction.

---

## **Output Example**

### **Sample Console Output**
```plaintext
Accuracy: 92.13%
Predictions saved to Predictions.csv
```

### **Sample Rows from Predictions.csv**
| Image ID        | Actual Label | Predicted Label |
|------------------|--------------|-----------------|
| cat.1001.jpg     | cat          | cat             |
| dog.2002.jpg     | dog          | dog             |
| cat.1023.jpg     | cat          | dog             |

---

## **Limitations**
1. The model uses grayscale images, which may lose information from color.
2. The linear kernel in SVM might not capture non-linear patterns effectively.
3. Fixed image size (\(64 \times 64\)) may discard critical details.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
