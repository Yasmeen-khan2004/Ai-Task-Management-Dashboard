import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import os

st.title("🧠 AI Task Management Dashboard")

# Load processed data
data_path = 'data/processed_task_dataset.csv'
df = pd.read_csv(data_path)

st.header("Dataset Overview")
st.write(df.head())
st.write(f"Shape: {df.shape}")
st.write("Columns:", df.columns.tolist())

# Show stats
st.subheader("Summary Statistics")
st.write(df.describe())

# Show plots
st.subheader("Visualizations")
col1, col2 = st.columns(2)
with col1:
    st.image('completion_time_vs_priority.png', caption='Completion Time vs Priority')
    st.image('progress_distribution.png', caption='Progress Distribution')
with col2:
    st.image('user_sentiment_distribution.png', caption='User Sentiment Distribution')
    st.image('wordcloud.png', caption='Word Cloud')

# Upload new dataset
st.subheader("Upload New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.write(new_df.head())

# Download processed data
st.subheader("Download Processed Dataset")
with open(data_path, 'rb') as f:
    st.download_button('Download CSV', f, file_name='processed_task_dataset.csv')

# 🧠 Load model
model_path = 'models/priority_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)

    st.subheader("🔮 Predict Task Priority")
    user_input = st.text_area("Enter task description:")
    if st.button("Predict Priority"):
        if user_input.strip() == "":
            st.warning("Please enter a task description.")
        else:
            prediction = model.predict([user_input])[0]
            st.success(f"Predicted Priority: **{prediction}**")
else:
    st.error("Model not found. Please train the model first.")
