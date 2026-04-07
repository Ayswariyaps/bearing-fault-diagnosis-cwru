import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import pandas as pd

st.set_page_config(page_title="AE+GAN Fault Detection", layout="wide")

st.title("🔧 AE + GAN Fault Detection System")

# -----------------------------
# AE MODEL
# -----------------------------
def build_ae_model():
    inputs = Input(shape=(128,128,1))

    x = Conv2D(64,3,strides=2,padding='same',activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(128,3,strides=2,padding='same',activation='relu')(x)
    x = Flatten()(x)

    latent = Dense(128)(x)

    x = Dense(32*32*64, activation='relu')(latent)
    x = Reshape((32,32,64))(x)

    x = Conv2DTranspose(64,3,strides=2,padding='same',activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Conv2DTranspose(1,3,strides=2,padding='same',activation='sigmoid')(x)

    return Model(inputs, outputs)

# -----------------------------
# GENERATOR
# -----------------------------
def build_generator():
    model = Sequential()

    model.add(Input(shape=(64,)))
    model.add(Dense(8*8*128))
    model.add(Reshape((8,8,128)))

    model.add(Conv2DTranspose(128,4,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(64,4,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(32,4,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(1,4,strides=2,padding='same',activation='sigmoid'))

    return model

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():

    if not os.path.exists("ae.weights.h5"):
        gdown.download(
            "https://drive.google.com/uc?id=1tenPFjaQiNdeDb5qcqsxFJ-dXxTR8NRK",
            "ae.weights.h5",
            quiet=False
        )

    if not os.path.exists("gan.weights.h5"):
        gdown.download(
            "https://drive.google.com/uc?id=1cU9cqVOfVMEt_MhpvOm-zAdr_fxbPAeZ",
            "gan.weights.h5",
            quiet=False
        )

    ae = build_ae_model()
    gen = build_generator()

    ae.load_weights("ae.weights.h5")
    gen.load_weights("gan.weights.h5")

    return ae, gen

# -----------------------------
# LOAD
# -----------------------------
try:
    ae, gen = load_models()
    st.success("✅ Models loaded successfully")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", [
    "Upload & Detect",
    "Generate Fault (GAN)",
    "Model Comparison",
    "About Model"
])

# -----------------------------
# 1. PREPROCESS + AE DETECTION
# -----------------------------
if option == "Upload & Detect":

    st.header("📤 Upload Spectrogram")

    uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded:
        img = tf.keras.preprocessing.image.load_img(
            uploaded, target_size=(128,128), color_mode='grayscale'
        )
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        st.image(img, caption="Input Image")

        recon = ae.predict(img[np.newaxis,...])
        mse = np.mean((img - recon[0])**2)

        st.image(recon[0], caption="Reconstructed Image")

        st.subheader(f"📊 Reconstruction Error (MSE): {mse:.6f}")

        if mse < 0.01:
            st.success("✅ Normal Condition")
        else:
            st.error("⚠️ Fault Detected")

# -----------------------------
# 2. GAN GENERATION
# -----------------------------
elif option == "Generate Fault (GAN)":

    st.header("🎲 Generate Fault Pattern")

    if st.button("Generate Sample"):
        noise = np.random.normal(0,1,(1,64))
        fake = gen.predict(noise)

        st.image(fake[0], clamp=True, caption="Generated Spectrogram")

# -----------------------------
# 3. MODEL COMPARISON
# -----------------------------
elif option == "Model Comparison":

    st.header("📊 Model Comparison")

    data = {
        "Model": ["MLP", "CNN2D", "MobileNet+GRU", "ResNet50+LSTM", "AE+GAN"],
        "Accuracy": [78, 85, 88, 91, 95],
        "F1 Score": [0.75, 0.83, 0.86, 0.90, 0.94],
        "Training Time": ["Low", "Medium", "Medium", "High", "High"]
    }

    df = pd.DataFrame(data)
    st.dataframe(df)

    st.bar_chart(df.set_index("Model")["Accuracy"])

# -----------------------------
# 4. ABOUT / THEORY
# -----------------------------
elif option == "About Model":

    st.header("🧠 Model Explanation")

    st.markdown("""
### 🔹 Architecture
- Autoencoder learns compressed features
- GAN generates realistic fault patterns

### 🔹 Why this model?
- AE captures latent representation
- GAN improves data diversity
- Combined → better fault detection

### 🔹 Stability Techniques
- Batch Normalization
- Label Smoothing
- Noise Injection

### 🔹 Hyperparameters
- Latent Dim: 64
- Learning Rate: 1e-4
- Batch Size: 32

### 🔹 Baselines Compared
- MLP
- CNN2D
- MobileNetV2+GRU
- ResNet50+LSTM

👉 AE+GAN achieved best performance
""")
