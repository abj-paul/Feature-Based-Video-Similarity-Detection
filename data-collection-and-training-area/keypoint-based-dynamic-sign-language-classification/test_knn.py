import numpy as np
import joblib
from load_data import __load_single_sample

classes = ['Dhaka', 'Zuddho', 'Korat', 'Chittagong', 'Jela', 'Bivag', 'Gari', 'Taka', 'Phasi', 'Grephtar', 'Joma', 'Rong', 'Tahobil', 'Akashi', 'Kuthar', 'Guitar', 'Faridpur', 'Shotru', 'Sobuj', 'Jel']
knn_model = joblib.load("knn_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

def predict(data):
    single_sample = __load_single_sample(data)
    reshaped_sample = single_sample.reshape(1, -1)

    predicted_class = knn_model.predict(reshaped_sample)[0]
    print(knn_model.predict(reshaped_sample))
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    print(f"Predicted Label: {predicted_label}")

predict("../data/Korat/sample0")
predict("../data/Jela/sample10")
predict("../data/Dhaka/sample0")