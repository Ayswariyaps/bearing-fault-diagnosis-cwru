import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fault Detection", layout="wide")

st.title("🔧 Bearing Fault Detection (AE + GAN)")

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
# GAN GENERATOR
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

ae, gen = load_models()
st.success("✅ Models Loaded")

# -----------------------------
# SIDEBAR
# -----------------------------
option = st.sidebar.radio("Select Option", [
    "Upload .mat & Detect Fault",
    "Generate Fault (GAN)"
])

# -----------------------------
# MAT → SPECTROGRAM (FIXED)
# -----------------------------
def mat_to_spectrogram(file):

    mat = scipy.io.loadmat(file)

    # Remove metadata keys
    keys = [k for k in mat.keys() if not k.startswith('__')]

    if len(keys) == 0:
        raise Exception("No usable signal found in .mat file")

    # Auto select signal
    signal = mat[keys[0]].squeeze()

    st.write(f"Detected Signal Key: {keys[0]}")

    # Generate spectrogram
    fig, ax = plt.subplots()
    ax.specgram(signal, Fs=12000)
    ax.axis('off')

    fig.canvas.draw()

    # ✅ FIXED (no tostring_rgb error)
    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3]

    plt.close(fig)

    # Preprocess
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (128,128))
    img = img.numpy() / 255.0

    return img

# -----------------------------
# 1. UPLOAD + DETECT
# -----------------------------
if option == "Upload .mat & Detect Fault":

    st.header("📂 Upload MATLAB File")

    uploaded = st.file_uploader("Upload .mat file", type=["mat"])

    if uploaded:
        try:
            img = mat_to_spectrogram(uploaded)

            st.image(img, caption="Generated Spectrogram")

            # AE Reconstruction
            recon = ae.predict(img[np.newaxis,...])
            mse = np.mean((img - recon[0])**2)

            st.subheader(f"Reconstruction Error: {mse:.6f}")

            if mse < 0.01:
                st.success("✅ Normal Condition")
            else:
                st.error("⚠️ Fault Detected")

        except Exception as e:
            st.error(str(e))

# -----------------------------
# 2. GAN GENERATION
# -----------------------------
elif option == "Generate Fault (GAN)":

    st.header("🎲 Generate Fault Pattern")

    if st.button("Generate"):

        noise = np.random.normal(0,1,(1,64))
        fake = gen.predict(noise)

        # Fix gray output
        img = fake[0].squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='inferno')
        ax.axis('off')

        st.pyplot(fig)
