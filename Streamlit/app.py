import streamlit as st
import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import librosa.display
import matplotlib.pyplot as plt
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="Deepfake Audio Detection", page_icon="🎵", layout="centered")

# Define class names globally
class_names = ['real', 'fake']

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
    <style>
        /* Global background color */
        .main {
            background: linear-gradient(135deg, #04013b 0%, #121212 100%);  /* Dark Gradient Background */
            color: white;
        }

        /* Centering the title and applying glowing effect */
        .title {
            color: #03fce8;  /* Neon Blue */
            font-weight: bold;
            font-size: 50px;  /* Increase font size for better visibility */
            text-align: center;
            text-shadow: 0 0 10px #03fce8, 0 0 20px #03fce8, 0 0 30px #03fce8, 0 0 40px #03fce8, 0 0 50px #03fce8, 0 0 75px #03fce8; /* Glowing effect */
        }

        /* Button Styling */
        .stButton button {
            background-color: #03fce8;  /* Neon Blue */
            color: white;
            font-size: 18px;
            border-radius: 12px;
            padding: 15px 30px;
            width: 100%;
            margin-top: 10px;
        }

        /* Button hover effect */
        .stButton button:hover {
            background-color: #1E90FF;  /* Lighter Neon Blue */
            color: black;
        }

        /* File Uploader Styling */
        .stFileUploader {
            background-color: #1e1e2f;
            border: 2px dashed #03fce8;  /* Neon Blue dashed border */
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
        }

        /* Custom font for the file uploader */
        .stFileUploader button {
            background-color: #03fce8;
            color: white;
            font-size: 16px;
        }

        /* Remove Streamlit default sidebar */
        .css-1d391kg {
            display: none;
        }

        /* Navigation Bar */
        .css-1q1gdr2 {
            display: flex;
            justify-content: space-between;
            background-color: #04013b;  /* Deep Purple Background */
            padding: 10px 30px;
        }

        .css-1q1gdr2 > div {
            color: #03fce8;
            font-size: 18px;
        }

        .stSidebar {
            display: none; /* Hide sidebar */
        }

        /* Container width for images */
        .stImage img {
            width: 100% !important; /* Full width image */
        }

        .stSpinner {
            color: #03fce8; /* Spinner color */
        }

        /* Adjust layout for navigation */
        .stApp {
            margin-top: 0px;
        }

        /* Custom prediction styling */
        .prediction-box {
            padding: 20px;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: white;
            border-radius: 10px;
            margin-top: 20px;
        }

        /* Red background for fake predictions */
        .fake {
            background-color: #FF4C4C; /* Red background for fake */
        }

        /* Green background for real predictions */
        .real {
            background-color: #4CAF50; /* Green background for real */
        }

    </style>
""", unsafe_allow_html=True)

# ------------------ SAVE UPLOADED FILE ------------------ #
def save_file(sound_file):
    with open(os.path.join('audio_files/', sound_file.name), 'wb') as f:
        f.write(sound_file.getbuffer())
    return sound_file.name

# ------------------ CREATE SPECTROGRAM ------------------ #
def create_spectrogram(sound):
    audio_file = os.path.join('audio_files/', sound)
    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_ms, sr=sr)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('melspectrogram.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    image_data = load_img('melspectrogram.png', target_size=(224, 224))
    st.image(image_data, caption="Generated Mel-Spectrogram", use_container_width=True)
    return image_data

# ------------------ PREDICTION ------------------ #
def predictions(image_data, model):
    img_array = np.array(image_data) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)
    return class_label, prediction

# ------------------ LIME EXPLAINABILITY ------------------ #
def lime_predict(image_data, model):
    img_array = np.array(image_data) / 255.0
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array.astype('double'), model.predict, hide_color=0, num_samples=1000)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    temp, mask = explanation.get_image_and_mask(np.argmax(model.predict(np.expand_dims(img_array, 0))), positive_only=False, num_features=8, hide_rest=False)

    axs[0].imshow(image_data)
    axs[0].set_title("Original Spectrogram")
    axs[1].imshow(mark_boundaries(temp, mask))
    axs[1].set_title("LIME Explanation")

    plt.tight_layout()
    st.pyplot(fig)

# ------------------ GRAD-CAM ------------------ #
def grad_predict(image_data, model, preds, class_idx):
    img_array = img_to_array(image_data)
    x = np.expand_dims(img_array, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)

    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    last_conv_layer = vgg_model.get_layer('block5_conv3')
    grad_model = tf.keras.models.Model([vgg_model.inputs], [last_conv_layer.output, vgg_model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(x)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_array.astype('float32'), 0.6, heatmap.astype('float32'), 0.4, 0)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(image_data)
    axs[0].set_title("Original Spectrogram")
    axs[1].imshow(superimposed_img.astype('uint8'))
    axs[1].set_title("Grad-CAM Explanation")

    plt.tight_layout()
    st.pyplot(fig)

# ------------------ MAIN UI ------------------ #
def main():
    # Centered Title with the glowing effect
    st.markdown('<h1 class="title">Deepfake Audio Detection</h1>', unsafe_allow_html=True)
    st.markdown("Upload a .wav file to classify if it's *real* or *fake*.")

    uploaded_file = st.file_uploader("Upload Audio File", type="wav")

    if uploaded_file:
        with st.spinner("Saving audio..."):
            save_file(uploaded_file)

        st.audio(uploaded_file.read(), format='audio/wav')

        with st.spinner("Generating Spectrogram..."):
            spec = create_spectrogram(uploaded_file.name)

        with st.spinner("Loading Model & Making Prediction..."):
            model = tf.keras.models.load_model('saved_model/model')
            class_label, prediction = predictions(spec, model)

        # Create a styled prediction result box
        prediction_text = f"🎯 Prediction: The uploaded audio is *{class_names[class_label].upper()}*"
        if class_names[class_label] == 'fake':
            st.markdown(f'<div class="prediction-box fake">{prediction_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-box real">{prediction_text}</div>', unsafe_allow_html=True)

        if st.button("🔍 Show Explanations"):
            st.subheader("LIME Visualization")
            lime_predict(spec, model)

            st.subheader("Grad-CAM Visualization")
            grad_predict(spec, model, prediction, class_label)
    else:
        st.info("Please upload a .wav file to continue.")

# ------------------ ENTRY POINT ------------------ #
if __name__ == "__main__":
    main()
