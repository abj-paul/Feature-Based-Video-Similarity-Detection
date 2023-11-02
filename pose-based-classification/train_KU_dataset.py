import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Assuming you have loaded images and keypoints
# images shape: (1500, 4624, 3468, 3)
# keypoints shape: (1500, 1662)
def load_KU_dataset(SAMPLE_NUM):
    images = []
    # loading keypoints
    classes = [
        "2433", "2435", "2454", "2456", "2458", "2460-2479", "2462", "2464", "2466", "2468-2510", "2470", "2474", "2476-2477", "2480-2524-2525", "2486-2488-2487",
        "2434", "2453", "2455", "2457", "2459", "2461", "2463", "2465", "2467-2472", "2469", "2471", "2475", "2478", "2482", "2489"
    ]
    keypoints = []
    class_labels = []
    for class_index, class_name in enumerate(classes):
        for sample_number in range(1, SAMPLE_NUM):
            image = cv2.imread(f'data/{class_name}/keypoint_sample_{sample_number}.jpg')
            image = cv2.resize(image, (128, 128))
            images.append(image)
            keypoint = np.load(f'data/{class_name}/sample_{sample_number}.npy')
            keypoints.append(keypoint)
            class_labels.append(class_index)
    return images, np.array(keypoints), np.array(class_labels)

images, keypoints, labels = load_KU_dataset(40)
print(np.array(images).shape, keypoints.shape, np.array(labels).shape)

# Define the number of classes (replace with your actual number)
num_classes = len(labels)

# Resize the images to a consistent size (e.g., 128x128)
resized_images = np.array([cv2.resize(image, (128, 128)) for image in images])

# Flatten the resized images and keypoints
images_flattened = resized_images.reshape(resized_images.shape[0], -1)  # Reshape to (1500, 128*128*3)
keypoints_flattened = keypoints.reshape(keypoints.shape[0], -1)  # Reshape to (1500, 1662)

num_classes = len(labels)

images = np.array(images, dtype='float32') / 255.0
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1

# Create a CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Adjust the loss function as needed
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)  # Adjust the number of epochs and batch size as needed

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')
