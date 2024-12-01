import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image

# Function to load n images and infer labels from filenames
def load_first_n_images_with_filenames(folder, n=25000, image_size=(64, 64)):
    images = []
    labels = []
    filenames = []
    count = 0

    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
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
                filenames.append(filename)
                count += 1
                if count >= n:
                    break  # Stop after loading the first n images
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    
    return images, labels, filenames


train_folder = 'Data_set/train'
images, labels, filenames = load_first_n_images_with_filenames(train_folder, n=25000)

# Convert lists to arrays
X = np.array(images)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(X, y, filenames, test_size=0.2, random_state=42
r)

# Train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Map numeric predictions to string labels
label_map = {0: 'cat', 1: 'dog'}
y_pred_labels = [label_map[pred] for pred in y_pred]
y_test_labels = [label_map[actual] for actual in y_test]

# Create a DataFrame with filenames, actual labels, and predicted labels
results_df = pd.DataFrame({
    'Image ID': filenames_test,
    'Actual Label': y_test_labels,
    'Predicted Label': y_pred_labels
})

# Save results to a CSV file
results_df.to_csv('Predictions.csv', index=False)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Predictions saved to Predictions.csv")

