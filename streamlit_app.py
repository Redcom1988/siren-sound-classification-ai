import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import librosa

# Import from our utility module
from model_utils import NeuralNetwork, extract_features, read_wav, load_model_from_json

# Set page configuration
st.set_page_config(
    page_title="Siren Sound Classification",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Siren Sound Classification")
st.markdown("Upload an audio file to classify different types of siren sounds")

st.sidebar.header("Model Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)

@st.cache_resource
def load_model():
    try:
        model, norm_params, label_map = load_model_from_json('siren_ann_model.json')
        st.success("Model loaded successfully!")
        return model, norm_params, label_map
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def plot_waveform(y, sr):
    time = np.linspace(0, len(y) / sr, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode='lines', name='Waveform'))
    fig.update_layout(title="Audio Waveform", xaxis_title="Time (seconds)", yaxis_title="Amplitude", height=300)
    return fig

def main():
    model, norm_params, label_map = load_model()
    
    if model is None:
        st.stop()
    
    # Display model info
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"Architecture: {' → '.join(map(str, model.layer_sizes))}")
    st.sidebar.write(f"Classes: {list(label_map.keys())}")
    
    int_to_label = {v: k for k, v in label_map.items()}
    
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
    
    if uploaded_file is not None:
        st.subheader("📁 File Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("Processing audio and extracting features..."):
                # Use your exact feature extraction
                features = extract_features(tmp_file_path)
                features_array = np.array(features).reshape(1, -1)
                
                # Normalize features
                mean, std = norm_params
                features_normalized = (features_array - mean) / std
                
                # Make prediction
                predictions = model.predict_proba(features_normalized)
                predicted_class_idx = model.predict(features_normalized)[0]
                confidence = predictions[0][predicted_class_idx]
                predicted_class = int_to_label[predicted_class_idx]
            
            # Display results
            st.subheader("🎯 Classification Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", predicted_class)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            if confidence >= confidence_threshold:
                st.success(f"High confidence prediction: {predicted_class}")
            else:
                st.warning(f"Low confidence prediction. Consider manual review.")
            
            # Prediction probabilities
            st.subheader("📊 Class Probabilities")
            prob_df = pd.DataFrame({
                'Class': [int_to_label[i] for i in range(len(predictions[0]))],
                'Probability': predictions[0]
            }).sort_values('Probability', ascending=False)
            
            fig_bar = px.bar(prob_df, x='Class', y='Probability', title="Class Prediction Probabilities")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Audio playback
            st.subheader("🔊 Audio Analysis")
            st.audio(uploaded_file)
            
            # Waveform
            samples, sample_rate = read_wav(tmp_file_path)
            fig_wave = plot_waveform(samples, sample_rate)
            st.plotly_chart(fig_wave, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()