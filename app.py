import streamlit as st
from fastai.vision.all import load_learner, PILImage

# Load your trained model
model = load_learner('bear_classifier_model.pkl')

st.title("Bear Classifier")
st.write("This app classifies bears as grizzly, black, or teddy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    pred, pred_idx, probs = model.predict(img)
    st.write(f"Prediction: {pred}")
    st.write(f"Probability: {probs[pred_idx]:.4f}")