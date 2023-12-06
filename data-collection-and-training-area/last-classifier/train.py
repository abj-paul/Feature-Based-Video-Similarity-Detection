import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

X = joblib.load("X.joblib")
Y = joblib.load("Y.joblib")

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
Y_onehot = encoder.fit_transform(Y.reshape(-1, 1))
num_classes = sum([1 for _ in set(Y)])
print(f"num_classes={num_classes}")

# Define the CNN model
def create_model(input_shape):
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D((2, 2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))  # Assuming binary classification, adjust for your case
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Adjust for your problem type
                  metrics=['accuracy'])
    
    return model

# Reshape X to fit the Conv3D input shape
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4])

# Create the model
input_shape = (X.shape[1], X.shape[2], X.shape[3], X.shape[4])


# Step 2: Split the Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

model = create_model(input_shape)

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32)

# Save the trained model
model.save('video_classification_model.h5')


# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, Y_test)

# Print the test accuracy
print(f'Test accuracy: {test_acc}')