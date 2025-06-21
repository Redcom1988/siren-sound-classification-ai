import numpy as np
import json
import wave
import struct

# Copy the exact feature extraction functions from your notebook
def read_wav(filename):
    with wave.open(filename, 'rb') as wav_file:
        n_channels, sampwidth, framerate, n_frames, _, _ = wav_file.getparams()
        raw_data = wav_file.readframes(n_frames)
        fmt = "<" + "h" * (n_frames * n_channels)
        data = struct.unpack(fmt, raw_data)
        print(f"📂 Membaca file: {filename} (sample rate: {framerate} Hz, total frames: {n_frames})")
        return np.array(data), framerate

def zero_crossing_rate(samples):
    # Vectorized zero crossing rate calculation
    signs = np.sign(samples)
    sign_changes = np.diff(signs)
    zcr = np.count_nonzero(sign_changes) / len(samples)
    return zcr

def energy(samples):
    # Vectorized energy calculation
    en = np.mean(samples**2)
    return en

def fft(samples):
    # Use NumPy's FFT implementation
    fft_result = np.fft.fft(samples)
    magnitudes = np.abs(fft_result)
    return magnitudes

def dominant_freq(magnitudes, sample_rate):
    # Find dominant frequency using NumPy
    max_index = np.argmax(magnitudes)
    freq = max_index * sample_rate / len(magnitudes)
    return freq

def pre_emphasis(signal, coeff=0.97):
    # Vectorized pre-emphasis filter
    emphasized = np.zeros_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coeff * signal[:-1]
    return emphasized

def hamming_window(N):
    # Use NumPy's hamming window
    return np.hamming(N)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(n_filters, N_fft, sample_rate):
    # Vectorized mel filterbank creation
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    
    # Create mel points
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor(hz_points * N_fft / sample_rate).astype(int)
    
    # Create filter bank matrix
    filters = np.zeros((n_filters, N_fft // 2))
    
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        
        # Left slope
        for j in range(left, center):
            if j < filters.shape[1]:
                filters[i - 1, j] = (j - left) / (center - left)
        
        # Right slope
        for j in range(center, right):
            if j < filters.shape[1]:
                filters[i - 1, j] = (right - j) / (right - center)
    
    return filters

def apply_filterbanks(magnitudes, filters):
    # Vectorized filterbank application
    # Ensure magnitudes length matches filter bank width
    mag_len = min(len(magnitudes), filters.shape[1])
    magnitudes_truncated = magnitudes[:mag_len]
    filters_truncated = filters[:, :mag_len]
    
    # Apply filters using matrix multiplication
    energies = np.dot(filters_truncated, magnitudes_truncated)
    # Add small epsilon to avoid log(0)
    energies = np.log(energies + 1e-10)
    return energies

def dct(signal):
    # Vectorized DCT implementation
    N = len(signal)
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    
    # DCT matrix
    dct_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    result = np.dot(dct_matrix, signal)
    return result

def mfcc(samples, sample_rate, num_filters=26, num_coeffs=13):
    emphasized = pre_emphasis(samples)
    
    # Windowing
    frame_size = int(0.025 * sample_rate)
    frame = emphasized[:frame_size]
    hamming = hamming_window(len(frame))
    windowed = frame * hamming
    
    # FFT
    spectrum = fft(windowed)
    
    # Mel filterbank
    filters = mel_filterbank(num_filters, len(spectrum) * 2, sample_rate)
    
    # Apply filterbanks
    energies = apply_filterbanks(spectrum, filters)
    
    # DCT
    cepstrals = dct(energies)
    mfccs = cepstrals[:num_coeffs]
    
    return mfccs.tolist()

def extract_features(file_path):
    print(f"\n📥 Ekstraksi fitur dari: {file_path}")
    samples, sample_rate = read_wav(file_path)
    
    zcr = zero_crossing_rate(samples)
    en = energy(samples)
    
    # Use first 512 samples for FFT
    fft_samples = samples[:512] if len(samples) >= 512 else samples
    fft_mags = fft(fft_samples)
    dom_freq = dominant_freq(fft_mags, sample_rate)
    
    mfcc_feats = mfcc(samples, sample_rate)
    
    return [zcr, en, dom_freq] + mfcc_feats

# Pure NumPy KNN implementation matching your notebook
class KNNModel:
    def __init__(self, train_data=None, train_labels=None, label_map=None, k=3):
        self.train_data = train_data
        self.train_labels = train_labels
        self.label_map = label_map
        self.k = k

    def euclidean_distance(self, a, b):
        # Vectorized Euclidean distance
        return np.sqrt(np.sum((a - b)**2))

    def predict_single(self, test_point):
        # Vectorized distance calculation for all training points
        distances = np.array([self.euclidean_distance(test_point, td) for td in self.train_data])
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.train_labels[k_indices]
        
        # Return most common label
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def predict(self, X):
        """Predict for multiple samples"""
        if len(X.shape) == 1:
            # Single sample
            return np.array([self.predict_single(X)])
        else:
            # Multiple samples
            predictions = []
            for test_point in X:
                pred = self.predict_single(test_point)
                predictions.append(pred)
            return np.array(predictions)

    def predict_proba(self, X):
        """Return class probabilities"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Get unique classes from training labels
        unique_classes = np.unique(self.train_labels)
        probabilities = []
        
        for test_point in X:
            # Calculate distances to all training points
            distances = np.array([self.euclidean_distance(test_point, td) for td in self.train_data])
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.train_labels[k_indices]
            
            # Calculate probabilities as frequency of each class in k neighbors
            probs = np.zeros(len(unique_classes))
            for i, class_label in enumerate(unique_classes):
                probs[i] = np.sum(k_labels == class_label) / self.k
            
            probabilities.append(probs)
        
        return np.array(probabilities)

def load_knn_model_from_json(json_path='siren_knn_model.json'):
    """Load KNN model from JSON file - matching your notebook format"""
    try:
        with open(json_path, 'r') as f:
            model_data = json.load(f)
        
        # Extract data according to your notebook's save format
        train_data = np.array(model_data['data'])
        train_labels = np.array(model_data['labels'])
        label_map = model_data['label_map']
        k = model_data.get('k', 3)  # Default k=3 if not specified
        
        # Create KNN model
        model = KNNModel(
            train_data=train_data,
            train_labels=train_labels,
            label_map=label_map,
            k=k
        )
        
        print(f"✅ Successfully loaded KNN model with {len(train_data)} training samples, k={k}")
        return model, label_map
        
    except KeyError as e:
        raise Exception(f"Missing key in model file: {e}. Please check the model file structure.")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def save_knn_model_to_json(train_data, train_labels, label_map, k=3, filename='siren_knn_model.json'):
    """Save KNN model to JSON file - matching your notebook format"""
    model = {
        'data': train_data.tolist(),  # Convert numpy array to list for JSON serialization
        'labels': train_labels.tolist(),
        'label_map': label_map,
        'k': k
    }
    
    with open(filename, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"\n✅ Model disimpan sebagai: {filename}")