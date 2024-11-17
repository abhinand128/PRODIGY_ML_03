import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image

# Function to load images and infer labels from filenames
def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            # Derive the label from the filename
            if 'cat' in filename:
                label = 0  # Label 0 for cats
            elif 'dog' in filename:
                label = 1  # Label 1 for dogs
            else:
                continue
            
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(image_size)
                img_array = np.array(img).flatten()  # Flatten the image
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    
    return images, labels

# Path to the train folder
train_folder = '/home/abhi/Documents/intern_ml/PRODIGY_ML_03/Data_set/train'

# Load data and labels
images, labels = load_images_from_folder(train_folder)

# Convert lists to arrays
X = np.array(images)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = SVC(kernel='linear')  # You can try other kernels like 'rbf' for experimentation
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Create a CSV file with predictions
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv('classification_results.csv', index=False)

print("Predictions saved to classification_results.csv")
