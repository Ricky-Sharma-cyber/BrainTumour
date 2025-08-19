import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.utils import normalize

# Load your trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Set input image size
INPUT_SIZE = 64

# Streamlit app setup
st.title("🧠 Brain Tumor Image Classifier")
st.write("Upload an MRI image to predict if it shows a brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((INPUT_SIZE, INPUT_SIZE))
    img_array = np.array(img)
    input_img = np.expand_dims(img_array, axis=0)
    input_img = normalize(input_img, axis=1)

    # Predict
    output = model.predict(input_img)
    predicted_class = np.argmax(output)
     # Display result
    diagnosis = "🟢 No Tumor Detected" if predicted_class == 0 else "🔴 Tumor Detected"
    st.subheader(f"Prediction: {diagnosis}")
    st.write(f"Confidence: {output[0][predicted_class]:.2f}")

# import streamlit as st
# import numpy as np
# import cv2
# import joblib  # or pickle
# from PIL import Image

# # Load pre-trained scikit-learn model
# model = joblib.load("tumor_classifier.pkl")

# # Image settings
# INPUT_SIZE = (64, 64)

# # UI setup
# st.title("🧠 Brain Tumor Classifier (No TF/Keras)")
# st.write("Upload an MRI scan image to detect a tumor.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("L")  # Grayscale
#     st.image(image, caption="Uploaded MRI", use_column_width=True)

#     # Preprocess
#     image = image.resize(INPUT_SIZE)
#     img_array = np.array(image).flatten() / 255.0  # Normalize
#     img_array = img_array.reshape(1, -1)

#     # Predict
#     prediction = model.predict(img_array)[0]
#     confidence = model.predict_proba(img_array)[0][prediction]

#     # Output
#     result = "🟢 No Tumor" if prediction == 0 else "🔴 Tumor Detected"
#     st.subheader(f"Prediction: {result}")
#     st.write(f"Confidence: {confidence:.2f}")
