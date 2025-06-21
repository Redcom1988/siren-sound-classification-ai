import numpy as np
import json
import wave
import struct
from scipy.fft import fft as scipy_fft

# Copy the exact classes and functions from your notebook
class NeuralNetwork:
    def __init__(self, layer_sizes=None, weights=None, biases=None, dropout_rates=None, lambd=0.001, input_size=None, hidden_layers=None, output_size=None):
        if layer_sizes is not None:
            # Loading from saved model
            self.layer_sizes = layer_sizes
            self.num_layers = len(layer_sizes) - 1
            self.weights = [np.array(w) for w in weights]
            self.biases = [np.array(b) for b in biases]
            self.dropout_rates = dropout_rates
            self.lambd = lambd
        else:
            # Creating new model (for training)
            self.layer_sizes = [input_size] + hidden_layers + [output_size]
            self.num_layers = len(self.layer_sizes) - 1
            self.weights = []
            self.biases = []
            self.dropout_rates = dropout_rates or [0.0] * self.num_layers
            self.lambd = lambd

    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X, training=False):
        activations = [X]
        
        # Forward through hidden layers
        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.leaky_relu(z)
            activations.append(a)

        # Output layer with softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = self.softmax(z)
        activations.append(a)

        return activations[-1]

    def predict(self, X):
        y_pred = self.forward(X, training=False)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        return self.forward(X, training=False)

# Feature extraction functions from your notebook
def read_wav(filename):
    with wave.open(filename, 'rb') as wav_file:
        n_channels, sampwidth, framerate, n_frames, _, _ = wav_file.getparams()
        raw_data = wav_file.readframes(n_frames)
        fmt = "<" + "h" * (n_frames * n_channels)
        data = struct.unpack(fmt, raw_data)
        return np.array(data), framerate

def zero_crossing_rate(samples):
    signs = np.sign(samples)
    sign_changes = np.diff(signs)
    zcr = np.count_nonzero(sign_changes) / len(samples)
    return zcr

def energy(samples):
    en = np.mean(samples**2)
    return en

def fft(samples):
    fft_result = np.fft.fft(samples)
    magnitudes = np.abs(fft_result)
    return magnitudes

def dominant_freq(magnitudes, sample_rate):
    max_index = np.argmax(magnitudes)
    freq = max_index * sample_rate / len(magnitudes)
    return freq

def pre_emphasis(signal, coeff=0.97):
    emphasized = np.zeros_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coeff * signal[:-1]
    return emphasized

def hamming_window(N):
    return np.hamming(N)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(n_filters, N_fft, sample_rate):
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor(hz_points * N_fft / sample_rate).astype(int)
    
    filters = np.zeros((n_filters, N_fft // 2))
    
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        
        for j in range(left, center):
            if j < filters.shape[1]:
                filters[i - 1, j] = (j - left) / (center - left)
        
        for j in range(center, right):
            if j < filters.shape[1]:
                filters[i - 1, j] = (right - j) / (right - center)
    
    return filters

def apply_filterbanks(magnitudes, filters):
    mag_len = min(len(magnitudes), filters.shape[1])
    magnitudes_truncated = magnitudes[:mag_len]
    filters_truncated = filters[:, :mag_len]
    
    energies = np.dot(filters_truncated, magnitudes_truncated)
    energies = np.log(energies + 1e-10)
    return energies

def dct(signal):
    N = len(signal)
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    
    dct_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    result = np.dot(dct_matrix, signal)
    return result

def mfcc(samples, sample_rate, num_filters=26, num_coeffs=13):
    emphasized = pre_emphasis(samples)
    
    frame_size = int(0.025 * sample_rate)
    frame = emphasized[:frame_size]
    hamming = hamming_window(len(frame))
    windowed = frame * hamming
    
    spectrum = fft(windowed)
    
    filters = mel_filterbank(num_filters, len(spectrum) * 2, sample_rate)
    
    energies = apply_filterbanks(spectrum, filters)
    
    cepstrals = dct(energies)
    mfccs = cepstrals[:num_coeffs]
    
    return mfccs.tolist()

def extract_features(file_path):
    samples, sample_rate = read_wav(file_path)
    
    zcr = zero_crossing_rate(samples)
    en = energy(samples)
    
    fft_samples = samples[:512] if len(samples) >= 512 else samples
    fft_mags = fft(fft_samples)
    dom_freq = dominant_freq(fft_mags, sample_rate)
    
    mfcc_feats = mfcc(samples, sample_rate)
    
    return [zcr, en, dom_freq] + mfcc_feats

def load_model_from_json(json_path='siren_ann_model.json'):
    # Load the trained model from JSON file
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    model = NeuralNetwork(
        layer_sizes=model_data['layer_sizes'],
        weights=model_data['weights'],
        biases=model_data['biases'],
        dropout_rates=model_data['dropout_rates'],
        lambd=model_data['lambd']
    )
    
    normalization_params = (
        np.array(model_data['normalization']['mean']),
        np.array(model_data['normalization']['std'])
    )
    
    label_map = model_data['label_map']
    
    return model, normalization_params, label_map