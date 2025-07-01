# app.py

import streamlit as st
from PIL import Image as PILImage
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle

# --------- Load Models ----------
cnn_model = load_model("models/cnn_xray_model.h5")

with open('models/logistic_regression_model.pkl', 'rb') as file:
    clf = pickle.load(file)

with open('models/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# --------- Load BERT Tokenizer and Model ----------
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# --------- Helper Functions ----------

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.numpy()

def predict_symptoms(symptom_input):
    emb = get_bert_embedding(symptom_input)
    pred_disease_encoded = clf.predict(emb)[0]
    pred_disease = le.inverse_transform([pred_disease_encoded])[0]
    return pred_disease

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = cnn_model.predict(img_array)
    class_idx = np.argmax(prediction)
    classes = ['COVID', 'NORMAL', 'VIRAL_PNEUMONIA']
    return classes[class_idx]

# --------- Streamlit UI Starts Here ----------

st.set_page_config(page_title="MediMind 2.0")
st.title("🩺 MediMind 2.0 - AI Healthcare Recommender System")

st.subheader("📝 Enter Symptoms:")
symptoms = st.text_input("Type symptoms separated by commas (e.g., fever, cough, headache)")

st.subheader("📤 Upload Chest X-ray (optional):")
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

# --------- Predict Button ----------
if st.button("🔍 Predict"):
    if symptoms:
        st.markdown("#### 🔍 NLP-based Prediction")
        predicted_disease = predict_symptoms(symptoms)
        st.success(f"Predicted Disease: {predicted_disease}")

    if uploaded_file:
        with open("temp_xray.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image("temp_xray.png", caption="Uploaded X-ray", width=250)
        st.markdown("#### 📸 CNN-based Prediction")
        predicted_class = predict_image("temp_xray.png")
        st.success(f"Predicted X-ray Class: {predicted_class}")

Move app.py to root
