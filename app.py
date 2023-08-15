import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model

model = load_model('model4.h5')

company_name = "HEVA"
logo_path = "Logo.png"

col1, col2 = st.columns([0.2, 1])

# Add the logo to the first column
col1.image(logo_path, width=80)

# Add the company name to the second column
col2.title(company_name)

st.title('Multi EEG Based Classification of Schizophrenia, Perkinsons, Frontotemporal Dementia, Depression and Alzhiemers Disease')
st.write("**Our Model Gives an Accuracy of 96.25%**")

uploaded_file = st.file_uploader("Choose a .edf of .fif file to upload", type=["edf","fif","set","bdf"], key="file")

if uploaded_file is not None:
    file_name = os.path.basename(uploaded_file.name)
    st.write(f"File Name: {file_name}")
    st.write(f"File Size: {uploaded_file.size} bytes")

    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension in [".edf"]:
        raw = mne.io.read_raw_edf(file_name,preload=True)
    elif file_extension in [".fif"]:
        raw = mne.io.read_raw_fif(file_name,preload=True)
    elif file_extension in [".bdf"]:
        raw = mne.io.read_raw_bdf(file_name,preload=True)
    elif file_extension in [".set"]:
        raw = mne.io.read_raw_eeglab(file_name,preload=True)

    raw.resample(sfreq=250)
    data = raw.get_data()
    df = pd.DataFrame(data)
    df = df.transpose()
    info = df.values
    info = (info - np.mean(info))/np.std(info)
    pca = PCA(n_components=10)
    pca.fit(info)
    info = pca.transform(info)

    X = info[0:15000]
    X = np.reshape(X,[1,15000,10])

    Y = model.predict(X)

    st.write(Y)

    st.write("**0 = Healthy**")
    st.write("**1 = Schizophrenia**")
    st.write("**2 = Perkinsons Disease**")
    st.write("**3 = Frontotemporal Dementia**")
    st.write("**4 = Depression**")
    st.write("**5 = Alzhiemers**")
    st.write("**All the results are out of 1**")
    os.remove(file_name)