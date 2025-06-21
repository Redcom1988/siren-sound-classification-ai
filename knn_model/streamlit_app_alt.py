import streamlit as st
import numpy as np
import os
from model_utils_knn import load_knn_model_from_json, extract_features
import tempfile

# Set page config
st.set_page_config(
    page_title="Siren Sound Classification - KNN",
    page_icon="🚨",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_map' not in st.session_state:
    st.session_state.label_map = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_model():
    """Load the KNN model"""
    try:
        json_path = 'siren_knn_model.json'
        
        if os.path.exists(json_path):
            st.session_state.model, st.session_state.label_map = load_knn_model_from_json(json_path)
            st.session_state.model_loaded = True
            return True, f"KNN model loaded successfully from {json_path}"
        else:
            return False, "No KNN model file found. Please ensure 'siren_knn_model.json' exists in the current directory."
    
    except Exception as e:
        # Reset session state on error
        st.session_state.model = None
        st.session_state.label_map = None
        st.session_state.model_loaded = False
        return False, f"Error loading KNN model: {str(e)}"

def predict_audio(audio_file):
    """Predict the class of an audio file"""
    if not st.session_state.model_loaded:
        return None, "Model not loaded"
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract features
        features = extract_features(tmp_path)
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = st.session_state.model.predict(features_array)[0]
        probabilities = st.session_state.model.predict_proba(features_array)[0]
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Get class names
        if st.session_state.label_map:
            # Create reverse mapping from label_map
            class_names = {v: k for k, v in st.session_state.label_map.items()}
            predicted_class = class_names.get(prediction, f"Class {prediction}")
        else:
            predicted_class = f"Class {prediction}"
        
        # Create probability dictionary
        prob_dict = {}
        if st.session_state.label_map:
            for class_name, class_idx in st.session_state.label_map.items():
                if class_idx < len(probabilities):
                    prob_dict[class_name] = probabilities[class_idx]
        else:
            for i, prob in enumerate(probabilities):
                prob_dict[f"Class {i}"] = prob
        
        return predicted_class, prob_dict
    
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None, f"Error during prediction: {str(e)}"

def main():
    st.title("🚨 Siren Sound Classification - KNN Model")
    st.markdown("Upload an audio file to classify the type of siren sound using K-Nearest Neighbors algorithm.")
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        
        # Model loading section
        if st.button("🔄 Load KNN Model", type="primary"):
            with st.spinner("Loading KNN model..."):
                success, message = load_model()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Display model status
        if st.session_state.model_loaded:
            st.success("✅ KNN Model Loaded")
            
            # Model details - fixed to use correct KNNModel attributes
            if hasattr(st.session_state.model, 'k'):
                st.info(f"**K (neighbors):** {st.session_state.model.k}")
            
            if hasattr(st.session_state.model, 'train_data'):
                st.info(f"**Training samples:** {len(st.session_state.model.train_data)}")
            
            if st.session_state.label_map:
                st.info(f"**Classes:** {len(st.session_state.label_map)}")
                with st.expander("View Classes"):
                    for class_name, class_idx in st.session_state.label_map.items():
                        st.write(f"• {class_name} (ID: {class_idx})")
        else:
            st.warning("⚠️ Model not loaded")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Audio Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload a WAV, MP3, FLAC, or M4A file"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"📁 File uploaded: {uploaded_file.name}")
            st.info(f"📏 File size: {uploaded_file.size:,} bytes")
            
            # Audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Prediction button
            if st.button("🔍 Classify Audio", type="primary", disabled=not st.session_state.model_loaded):
                if not st.session_state.model_loaded:
                    st.error("Please load the KNN model first!")
                else:
                    with st.spinner("Analyzing audio with KNN..."):
                        predicted_class, probabilities = predict_audio(uploaded_file)
                        
                        if predicted_class:
                            st.success(f"🎯 **Prediction:** {predicted_class}")
                            
                            # Display probabilities
                            st.subheader("📊 Class Probabilities")
                            
                            # Sort probabilities by confidence
                            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                            
                            for class_name, prob in sorted_probs:
                                confidence = prob * 100
                                st.write(f"**{class_name}:** {confidence:.2f}%")
                                st.progress(prob)
                        else:
                            st.error(f"Prediction failed: {probabilities}")
    
    with col2:
        st.header("ℹ️ About KNN Model")
        st.markdown("""
        **K-Nearest Neighbors (KNN)** is a simple, instance-based learning algorithm that:
        
        - 🎯 Classifies samples based on the majority class of K nearest neighbors
        - 📏 Uses distance metrics to find similar audio samples
        - 🔢 No complex training phase - stores all training data
        - ⚡ Fast inference for small datasets
        
        **Features Used:**
        - Zero Crossing Rate (ZCR)
        - Energy
        - Dominant Frequency
        - MFCC coefficients (13 features)
        
        **Total Features:** 16
        """)
        
        # Feature extraction info
        with st.expander("🔧 Feature Details"):
            st.markdown("""
            **Zero Crossing Rate (ZCR):**
            - Measures how often the signal crosses zero
            - Indicates speech vs. music characteristics
            
            **Energy:**
            - Average power of the signal
            - Distinguishes loud vs. quiet sounds
            
            **Dominant Frequency:**
            - Most prominent frequency in the spectrum
            - Key identifier for different siren types
            
            **MFCC (Mel-Frequency Cepstral Coefficients):**
            - 13 coefficients representing spectral shape
            - Mimics human auditory perception
            - Standard in audio classification
            """)
    
    # Instructions
    st.header("📝 Instructions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Step 1: Load Model**
        - Click "Load KNN Model" in the sidebar
        - Ensure model file exists in the directory
        """)
    
    with col2:
        st.markdown("""
        **Step 2: Upload Audio**
        - Choose a WAV, MP3, FLAC, or M4A file
        - Listen to verify the audio
        """)
    
    with col3:
        st.markdown("""
        **Step 3: Classify**
        - Click "Classify Audio" button
        - View prediction and confidence scores
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🚨 Siren Sound Classification using K-Nearest Neighbors | "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()