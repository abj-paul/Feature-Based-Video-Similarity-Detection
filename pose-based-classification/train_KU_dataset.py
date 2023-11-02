import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_class_data(activities, SAMPLE_NUM):
    class_dataset = []
    class_labels = []
    for class_index, class_name in enumerate(activities):
        for sample_number in range(1, SAMPLE_NUM):
            keypoints = np.load(f'data/{class_name}/sample_{sample_number}.npy')
            class_dataset.append(keypoints)
            class_labels.append(class_index)
    return np.array(class_dataset), class_labels

directory_structure = [
    "2433", "2435", "2454", "2456", "2458", "2460-2479", "2462", "2464", "2466", "2468-2510", "2470", "2474", "2476-2477", "2480-2524-2525", "2486-2488-2487","2434", "2453", "2455", "2457", "2459", "2461", "2463", "2465", "2467-2472", "2469", "2471", "2475", "2478", "2482", "2489"
]

tags = directory_structure #["Camera", "miser", "Surprise"]
num_classes = len(tags)
keypoints_data, labels = load_class_data(directory_structure, 45)

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=num_classes)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(keypoints_data, labels, test_size=0.2, random_state=42)
#print(y_train.shape)

# Create a simple fully connected neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(keypoints_data.shape[1],)),  # Input shape depends on your keypoint data
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')  # Adjust 'num_classes' as needed
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Adjust the loss function as needed
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)  # Adjust the number of epochs and batch size as needed

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')
model.save("keypoints_based_sign_recognition_model.h5")  # Save it as an HDF5 file (recommended)

# Make predictions on new keypoints data
new_keypoints_data = keypoints_data[0]  # Replace with your new data
new_keypoints_data = new_keypoints_data.reshape(1, -1)  # Reshape to (1, num_features)
predictions = model.predict(new_keypoints_data)
print(predictions)
# The predictions can be used for classification.

