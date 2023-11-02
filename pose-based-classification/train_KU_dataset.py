import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from sklearn.model_selection import train_test_split
from collect_pose_data import get_image_paths_from_KU_dataset
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
            keypoint = np.load(f'data/{class_name}/sample_{sample_number}.npy')
            keypoints.append(keypoint)
            class_labels.append(class_index)


    directory_image_paths = get_image_paths_from_KU_dataset("MSLD/")
    for directory_name, folder_images in directory_image_paths:
        count = 0
        for index,image_path in enumerate(folder_images):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))
            images.append(image)
            count += 1
            if count >= SAMPLE_NUM-1:
                break

    return images, np.array(keypoints), np.array(class_labels)

images, keypoints, labels = load_KU_dataset(10)
print(np.array(images).shape, keypoints.shape, np.array(labels).shape)

# Define the number of classes (replace with your actual number)
num_classes = len(labels)

# Resize the images to a consistent size (e.g., 128x128)
resized_images = np.array([cv2.resize(image, (128, 128)) for image in images])

# Flatten the resized images and keypoints
images_flattened = resized_images.reshape(resized_images.shape[0], -1)  # Reshape to (1500, 128*128*3)
keypoints_flattened = keypoints.reshape(keypoints.shape[0], -1)  # Reshape to (1500, 1662)

# Split the data into training and testing sets
x_train_img, x_test_img, x_train_keypoints, x_test_keypoints, y_train, y_test = train_test_split(
    images_flattened, keypoints_flattened, labels, test_size=0.2, random_state=42
)


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the image processing part of the model
image_model = keras.Sequential([
    keras.layers.Input(shape=(images_flattened.shape[1],)),  # Adjust input shape based on the flattened image size
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu')
])

# Define the keypoints processing part of the model
keypoints_model = keras.Sequential([
    keras.layers.Input(shape=(keypoints_flattened.shape[1],)),  # Adjust input shape based on the flattened keypoints size
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu')
])

# Concatenate the outputs of both models
combined_model = keras.layers.concatenate([image_model.output, keypoints_model.output])

# Add additional layers for classification
x = keras.layers.Dense(64, activation='relu')(combined_model)
x = keras.layers.Dense(32, activation='relu')(x)
output = keras.layers.Dense(num_classes, activation='softmax')(x)

# Create the final model
model = keras.models.Model(inputs=[image_model.input, keypoints_model.input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([x_train_img, x_train_keypoints], y_train, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([x_test_img, x_test_keypoints], y_test)
print(f'Test accuracy: {test_accuracy}')

