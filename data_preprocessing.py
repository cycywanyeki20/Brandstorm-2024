import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Define the directory containing your images
image_dir = 'C:\\Users\\Cycy\\desktop\\hairtech\\images'

# Function to load and preprocess images
def load_and_preprocess_images(directory, image_size=(224, 224)):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            try:
                # Load and resize the image
                image = Image.open(image_path).resize(image_size)
                # Convert the image to numpy array and normalize pixel values
                image = np.array(image) / 255.0
                images.append(image)
                # Extract label from the filename or directory structure if available
                # You may need to customize this part based on your directory structure
                labels.append(filename.split('_')[0])  # Assuming filenames are like 'curly_001.jpg'

            except Exception as e:
                print(f"Error loading image: {e}")

    return np.array(images), np.array(labels)

# Load and preprocess images
X, y = load_and_preprocess_images(image_dir)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

