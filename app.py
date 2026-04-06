import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
import os
import numpy as np

st.title("AE + GAN Fault Detection")

# -----------------------------
# AE ARCHITECTURE
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
# DOWNLOAD + LOAD WEIGHTS
# -----------------------------
@st.cache_resource
def load_models():

    # AE weights
    if not os.path.exists("ae.weights.h5"):
        with st.spinner("Downloading AE weights..."):
            gdown.download(
                "https://drive.google.com/uc?id=1tenPFjaQiNdeDb5qcqsxFJ-dXxTR8NRK",
                "ae.weights.h5",
                quiet=False
            )

    # GAN weights
    if not os.path.exists("gan.weights.h5"):
        with st.spinner("Downloading GAN weights..."):
            gdown.download(
                "https://drive.google.com/uc?id=1cU9cqVOfVMEt_MhpvOm-zAdr_fxbPAeZ",
                "gan.weights.h5",
                quiet=False
            )

    # Build models
    ae = build_ae_model()
    gen = build_generator()

    # Load weights
    ae.load_weights("ae.weights.h5")
    gen.load_weights("gan.weights.h5")

    return ae, gen


# -----------------------------
# LOAD MODELS
# -----------------------------
try:
    ae, gen = load_models()
    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# -----------------------------
# GAN DEMO
# -----------------------------
st.subheader("Generate Fault Pattern (GAN)")

if st.button("Generate Sample"):
    noise = np.random.normal(0,1,(1,64))
    fake = gen.predict(noise)

    st.image(fake[0], clamp=True, caption="Generated Spectrogram")
