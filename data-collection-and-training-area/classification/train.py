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

# Step 2: Split the Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

# Step 4: Model Architecture
model = models.Sequential()
model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(16384, 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(21, activation='softmax'))

# Step 5: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Training
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1)

# Step 7: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc}')

# Step 8: Save the Model
model.save('video_classification_model.h5')
