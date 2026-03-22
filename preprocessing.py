"""
Converts real EMG to spinal bypass sensor format with signal processing

Features:
- Bandpass filtering
- Notch filter (50/60 Hz)
- RMS envelope extraction
- Wavelet denoising
- Feature scaling
- Data augmentation
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# Optional imports
try:
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Using basic preprocessing.")


class SignalProcessor:
    """
    Advanced signal processing for EMG data with enhanced denoising
    """
    
    def __init__(self, sampling_rate: int = 200):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def adaptive_filter(self, data: np.ndarray, noise_threshold: float = 2.0) -> np.ndarray:
        """
        Adaptive filtering based on signal-to-noise characteristics
        
        Args:
            data: (N, channels) EMG data
            noise_threshold: Threshold for noise detection (sigma)
        
        Returns:
            Filtered data
        """
        filtered = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Estimate noise level using MAD (Median Absolute Deviation)
            median = np.median(channel_data)
            mad = np.median(np.abs(channel_data - median))
            noise_level = mad * 1.4826  # Robust estimate of std
            
            # Adaptive threshold
            threshold = noise_threshold * noise_level
            
            # Identify noisy samples
            noise_mask = np.abs(channel_data - median) > threshold
            
            # Apply median filter to noisy regions
            if np.any(noise_mask):
                # Use rolling median for noisy segments
                window_size = min(11, len(channel_data) // 10)  # Adaptive window
                if window_size % 2 == 0:
                    window_size += 1  # Ensure odd
                
                from scipy.ndimage import median_filter
                filtered_channel = median_filter(channel_data, size=window_size)
                
                # Only replace noisy samples
                filtered[:, ch] = np.where(noise_mask, filtered_channel, channel_data)
            else:
                filtered[:, ch] = channel_data
        
        return filtered
    
    def wavelet_denoise(self, data: np.ndarray, wavelet: str = 'db4', level: int = 4) -> np.ndarray:
        """
        Wavelet denoising for EMG signals
        
        Args:
            data: (N, channels) EMG data
            wavelet: Wavelet type
            level: Decomposition level
        
        Returns:
            Denoised data
        """
        try:
            import pywt
        except ImportError:
            warnings.warn("pywt not available. Skipping wavelet denoising.")
            return data
        
        filtered = np.zeros_like(data)
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(channel_data, wavelet, level=level)
            
            # Threshold detail coefficients (soft thresholding)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust noise estimate
            
            for i in range(1, len(coeffs)):
                threshold = sigma * np.sqrt(2 * np.log(len(channel_data)))
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            
            # Reconstruction
            filtered[:, ch] = pywt.waverec(coeffs, wavelet)
        
        return filtered
    
    def baseline_correction(self, data: np.ndarray, method: str = 'median') -> np.ndarray:
        """
        Remove baseline drift
        
        Args:
            data: (N, channels) EMG data
            method: 'median' or 'polynomial'
        
        Returns:
            Baseline-corrected data
        """
        corrected = np.zeros_like(data)
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            if method == 'median':
                # Rolling median baseline
                window_size = max(51, len(channel_data) // 20)  # Adaptive window
                if window_size % 2 == 0:
                    window_size += 1
                
                baseline = self._rolling_median(channel_data, window_size)
                corrected[:, ch] = channel_data - baseline
                
            elif method == 'polynomial':
                # Polynomial detrending
                x = np.arange(len(channel_data))
                coeffs = np.polyfit(x, channel_data, deg=3)
                baseline = np.polyval(coeffs, x)
                corrected[:, ch] = channel_data - baseline
        
        return corrected
    
    def _rolling_median(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute rolling median"""
        if not SCIPY_AVAILABLE:
            # Simple implementation
            result = np.zeros_like(data)
            half_window = window_size // 2
            
            for i in range(len(data)):
                start = max(0, i - half_window)
                end = min(len(data), i + half_window + 1)
                result[i] = np.median(data[start:end])
            
            return result
        else:
            from scipy.ndimage import median_filter
            return median_filter(data, size=window_size)
    
    def outlier_removal(self, data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """
        Remove outliers using statistical methods
        
        Args:
            data: (N, channels) EMG data
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: Outlier threshold
        
        Returns:
            Data with outliers removed/replaced
        """
        cleaned = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            if method == 'iqr':
                # IQR method
                q1, q3 = np.percentile(channel_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers = (channel_data < lower_bound) | (channel_data > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((channel_data - np.mean(channel_data)) / np.std(channel_data))
                outliers = z_scores > threshold
            
            # Replace outliers with median of neighboring values
            if np.any(outliers):
                outlier_indices = np.where(outliers)[0]
                
                for idx in outlier_indices:
                    # Use median of surrounding values
                    start = max(0, idx - 5)
                    end = min(len(channel_data), idx + 6)
                    neighbor_median = np.median(channel_data[start:end])
                    cleaned[idx, ch] = neighbor_median
        
        return cleaned
    
    def bandpass_filter(self, 
                       data: np.ndarray,
                       lowcut: float = 20.0,
                       highcut: float = 450.0,
                       order: int = 4) -> np.ndarray:
        """
        Bandpass filter (removes low-freq drift and high-freq noise)
        
        Args:
            data: (N, channels) EMG data
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Filtered data
        """
        if not SCIPY_AVAILABLE:
            warnings.warn("scipy not available. Skipping bandpass filter.")
            return data
        
        # Adjust cutoff frequencies to respect Nyquist limit
        # Nyquist = sampling_rate / 2
        max_freq = self.nyquist * 0.95  # Use 95% of Nyquist to be safe
        
        # Clamp highcut to max allowable frequency
        if highcut >= max_freq:
            highcut = max_freq
            print(f"    ⚠ Highcut adjusted to {highcut:.0f} Hz (Nyquist limit)")
        
        # Ensure lowcut is reasonable
        if lowcut >= highcut:
            lowcut = highcut * 0.1  # 10% of highcut
            print(f"    ⚠ Lowcut adjusted to {lowcut:.0f} Hz")
        
        # Normalize frequencies
        low = lowcut / self.nyquist
        high = highcut / self.nyquist
        
        # Ensure normalized frequencies are in valid range (0, 1)
        if low <= 0 or low >= 1 or high <= 0 or high >= 1:
            warnings.warn(f"Invalid filter frequencies. Skipping bandpass filter.")
            return data
        
        # Design Butterworth filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
        
        return filtered
    
    def notch_filter(self,
                    data: np.ndarray,
                    freq: float = 60.0,
                    quality: float = 30.0) -> np.ndarray:
        """
        Notch filter (removes power line interference)
        
        Args:
            data: (N, channels) EMG data
            freq: Frequency to remove (50 Hz EU, 60 Hz US)
            quality: Quality factor
        
        Returns:
            Filtered data
        """
        if not SCIPY_AVAILABLE:
            return data
        
        # Check if notch frequency is valid
        if freq >= self.nyquist:
            warnings.warn(f"Notch frequency {freq} Hz exceeds Nyquist {self.nyquist} Hz. Skipping notch filter.")
            return data
        
        # Design notch filter
        w0 = freq / self.nyquist
        
        # Ensure w0 is in valid range
        if w0 <= 0 or w0 >= 1:
            warnings.warn(f"Invalid notch frequency. Skipping notch filter.")
            return data
        
        b, a = signal.iirnotch(w0, quality)
        
        # Apply to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
        
        return filtered
    
    def rms_envelope(self, 
                    data: np.ndarray,
                    window_size: int = 50) -> np.ndarray:
        """
        Extract RMS envelope (amplitude modulation)
        
        Args:
            data: (N, channels) EMG data
            window_size: Window size in samples
        
        Returns:
            RMS envelope
        """
        N, channels = data.shape
        envelope = np.zeros_like(data)
        
        for ch in range(channels):
            # Squared signal
            squared = data[:, ch] ** 2
            
            # Moving average
            kernel = np.ones(window_size) / window_size
            if SCIPY_AVAILABLE:
                averaged = signal.convolve(squared, kernel, mode='same')
            else:
                # Simple moving average
                averaged = np.convolve(squared, kernel, mode='same')
            
            # Square root
            envelope[:, ch] = np.sqrt(averaged)
        
        return envelope
    
    def full_preprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline with enhanced denoising
        
        Steps:
        1. Baseline correction
        2. Outlier removal
        3. Bandpass filter (20-450 Hz)
        4. Notch filter (60 Hz)
        5. Adaptive filtering
        6. Wavelet denoising
        7. RMS envelope
        8. Normalization
        """
        print("  ⟳ Baseline correction...")
        data = self.baseline_correction(data, method='median')
        
        print("  ⟳ Outlier removal...")
        data = self.outlier_removal(data, method='iqr', threshold=1.5)
        
        print("  ⟳ Bandpass filtering (20-450 Hz)...")
        data = self.bandpass_filter(data)
        
        print("  ⟳ Notch filtering (60 Hz)...")
        data = self.notch_filter(data)
        
        print("  ⟳ Adaptive filtering...")
        data = self.adaptive_filter(data, noise_threshold=2.0)
        
        print("  ⟳ Wavelet denoising...")
        data = self.wavelet_denoise(data)
        
        print("  ⟳ Extracting RMS envelope...")
        data = self.rms_envelope(data)
        
        print("  ⟳ Normalizing...")
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
        
        return data


class SpinalBypassConverter:
    """
    Convert EMG data to spinal bypass sensor format
    
    Output sensors:
    - TMR: (N, 8) - Spinal nerve activity at L1-L4
    - sEMG: (N, 64) - Surface muscle activity
    - IMU: (N, 3) - Joint angles [hip, knee, ankle]
    """
    
    def __init__(self):
        self.processor = SignalProcessor()
    
    def convert(self, 
                emg_data: np.ndarray,
                sampling_rate: int = 200,
                preprocess: bool = True) -> Dict[str, np.ndarray]:
        """
        Convert EMG to spinal bypass format
        
        Args:
            emg_data: (N, channels) raw EMG
            sampling_rate: Sampling rate in Hz
            preprocess: Apply signal processing
        
        Returns:
            Dictionary with 'tmr', 'semg', 'imu'
        """
        print(f"\n{'='*70}")
        print("PREPROCESSING EMG → SPINAL BYPASS FORMAT")
        print(f"{'='*70}")
        print(f"Input: {emg_data.shape[0]:,} samples × {emg_data.shape[1]} channels")
        
        N, n_channels = emg_data.shape
        
        # Update sampling rate
        self.processor.sampling_rate = sampling_rate
        self.processor.nyquist = sampling_rate / 2
        
        # Preprocess if requested
        if preprocess:
            print("\nStep 1: Signal Processing")
            emg_processed = self.processor.full_preprocessing(emg_data)
        else:
            print("\nStep 1: Skipping preprocessing (using raw data)")
            emg_processed = emg_data
        
        print("\nStep 2: Generating Spinal Bypass Sensors")
        
        # ===== sEMG: Surface EMG (64 channels) =====
        print("  → sEMG (64 channels)...")
        
        if n_channels >= 64:
            # Downsample channels
            indices = np.linspace(0, n_channels-1, 64, dtype=int)
            semg = emg_processed[:, indices]
        else:
            # Upsample by repeating and interpolating
            repeat_factor = (64 // n_channels) + 1
            semg = np.tile(emg_processed, (1, repeat_factor))[:, :64]
        
        # Normalize
        semg = (semg - np.mean(semg, axis=0)) / (np.std(semg, axis=0) + 1e-6)
        
        # ===== TMR: Spinal nerve activity (8 sensors) =====
        print("  → TMR (8 sensors at L1-L4)...")
        
        tmr = np.zeros((N, 8))
        
        # Divide sEMG into regions (simulate nerve root activity)
        # L1/L2 (hip flexors) = sEMG channels 0-16
        # L3/L4 (knee extensors) = sEMG channels 16-32
        # L5 (ankle) = sEMG channels 48-64
        
        tmr[:, 0] = np.mean(semg[:, 0:8], axis=1)    # L1 left
        tmr[:, 1] = np.mean(semg[:, 8:16], axis=1)   # L1 right
        tmr[:, 2] = np.mean(semg[:, 0:8], axis=1) * 1.1   # L2 left
        tmr[:, 3] = np.mean(semg[:, 8:16], axis=1) * 0.9  # L2 right
        tmr[:, 4] = np.mean(semg[:, 16:24], axis=1)  # L3 left
        tmr[:, 5] = np.mean(semg[:, 24:32], axis=1)  # L3 right
        tmr[:, 6] = np.mean(semg[:, 16:24], axis=1)  # L4 left
        tmr[:, 7] = np.mean(semg[:, 24:32], axis=1)  # L4 right
        
        # Enhanced TMR denoising and balancing
        print("  → TMR denoising and balancing...")
        tmr = self._denoise_tmr(tmr)
        
        # Add realistic TMR noise (reduced for better SNR)
        tmr += np.random.normal(0, 0.1, tmr.shape)  # Reduced from 0.2
        
        # Scale to realistic TMR range (0.1-5 nT)
        tmr = (tmr - tmr.min()) / (tmr.max() - tmr.min() + 1e-6)
        tmr = tmr * 4.0 + 0.5  # Scale to 0.5-4.5 nT
        
        # ===== IMU: Joint angles (3 DOF) =====
        print("  → IMU (hip, knee, ankle angles)...")
        
        imu = np.zeros((N, 3))
        
        # Hip angle: derived from L1/L2 activity
        hip_activity = np.mean(tmr[:, 0:4], axis=1)
        imu[:, 0] = hip_activity * 30 - 15  # Range: -15° to +45°
        
        # Knee angle: derived from L3/L4 activity
        knee_activity = np.mean(tmr[:, 4:8], axis=1)
        imu[:, 1] = 60 - knee_activity * 30  # Range: 0° to 60°
        
        # Ankle angle: derived from distal sEMG
        ankle_activity = np.mean(semg[:, 48:64], axis=1)
        imu[:, 2] = ankle_activity * 15 - 5  # Range: -20° to +10°
        
        # Smooth IMU signals (reduce jitter)
        if SCIPY_AVAILABLE:
            for i in range(3):
                imu[:, i] = gaussian_filter1d(imu[:, i], sigma=2.0)
        
        print(f"\n✓ Conversion complete:")
        print(f"  TMR:  {tmr.shape} (range: {tmr.min():.2f} to {tmr.max():.2f} nT)")
        print(f"  sEMG: {semg.shape} (normalized)")
        print(f"  IMU:  {imu.shape}")
        print(f"    Hip:   {imu[:,0].min():.1f}° to {imu[:,0].max():.1f}°")
        print(f"    Knee:  {imu[:,1].min():.1f}° to {imu[:,1].max():.1f}°")
        print(f"    Ankle: {imu[:,2].min():.1f}° to {imu[:,2].max():.1f}°")
        
        return {
            'tmr': tmr,
            'semg': semg,
            'imu': imu,
        }
    
    def _denoise_tmr(self, tmr: np.ndarray) -> np.ndarray:
        """
        Advanced TMR denoising and balancing
        
        Args:
            tmr: (N, 8) raw TMR data
        
        Returns:
            Denoised and balanced TMR data
        """
        denoised = tmr.copy()
        
        # 1. Temporal smoothing (reduce high-frequency noise)
        if SCIPY_AVAILABLE:
            for i in range(8):
                denoised[:, i] = gaussian_filter1d(denoised[:, i], sigma=1.0)
        
        # 2. Left-right balancing
        left_channels = [0, 2, 4, 6]   # L1, L2, L3, L4 left
        right_channels = [1, 3, 5, 7]  # L1, L2, L3, L4 right
        
        # Calculate imbalance factor
        left_mean = np.mean(denoised[:, left_channels], axis=1)
        right_mean = np.mean(denoised[:, right_channels], axis=1)
        
        imbalance_ratio = np.mean(left_mean) / (np.mean(right_mean) + 1e-10)
        
        # Apply balancing correction
        if imbalance_ratio > 1.2:  # Left side stronger
            # Boost right side
            correction_factor = min(imbalance_ratio * 0.1, 0.3)  # Max 30% boost
            denoised[:, right_channels] *= (1.0 + correction_factor)
        elif imbalance_ratio < 0.8:  # Right side stronger
            # Boost left side
            correction_factor = min((1.0 / imbalance_ratio) * 0.1, 0.3)
            denoised[:, left_channels] *= (1.0 + correction_factor)
        
        # 3. Cross-talk reduction (adjacent channel correlation)
        for i in range(0, 8, 2):  # Process each level pair
            left_ch = i
            right_ch = i + 1
            
            # Calculate correlation
            correlation = np.corrcoef(denoised[:, left_ch], denoised[:, right_ch])[0, 1]
            
            # If highly correlated (>0.8), reduce similarity
            if abs(correlation) > 0.8:
                # Orthogonalize signals slightly
                alpha = 0.1  # Orthogonalization strength
                mean_signal = (denoised[:, left_ch] + denoised[:, right_ch]) / 2
                diff_signal = (denoised[:, left_ch] - denoised[:, right_ch]) / 2
                
                denoised[:, left_ch] = mean_signal + alpha * diff_signal
                denoised[:, right_ch] = mean_signal - alpha * diff_signal
        
        # 4. Outlier removal for TMR
        for i in range(8):
            channel_data = denoised[:, i]
            
            # Use robust statistics
            median = np.median(channel_data)
            mad = np.median(np.abs(channel_data - median))
            
            # Identify outliers (beyond 3 MAD)
            outliers = np.abs(channel_data - median) > 3 * mad
            
            if np.any(outliers):
                # Replace with median
                denoised[outliers, i] = median
        
        return denoised


class DataAugmentor:
    """
    Data augmentation for improving ML generalization
    """
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def time_warp(data: np.ndarray, factor: float = 0.1) -> np.ndarray:
        """Time warping (speed up/slow down)"""
        N = len(data)
        new_N = int(N * (1 + np.random.uniform(-factor, factor)))
        
        # Resample
        indices = np.linspace(0, N-1, new_N)
        warped = np.zeros((new_N, data.shape[1]))
        
        for ch in range(data.shape[1]):
            warped[:, ch] = np.interp(indices, np.arange(N), data[:, ch])
        
        return warped
    
    @staticmethod
    def amplitude_scale(data: np.ndarray, factor: float = 0.2) -> np.ndarray:
        """Random amplitude scaling"""
        scale = 1 + np.random.uniform(-factor, factor)
        return data * scale
    
    @staticmethod
    def augment(data: np.ndarray, n_augmentations: int = 3) -> list:
        """
        Generate multiple augmented versions
        
        Returns:
            List of augmented data arrays
        """
        augmented = [data]  # Original
        
        for _ in range(n_augmentations):
            # Random combination of augmentations
            aug = data.copy()
            
            if np.random.random() > 0.5:
                aug = DataAugmentor.add_noise(aug)
            
            if np.random.random() > 0.5:
                aug = DataAugmentor.amplitude_scale(aug)
            
            # Note: time_warp changes length, so skip for now
            
            augmented.append(aug)
        
        return augmented


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              ADVANCED SIGNAL PREPROCESSING                           ║
║                                                                      ║
║        EMG → Spinal Bypass Sensor Format Conversion                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nFeatures:")
    print("  ✓ Bandpass filtering (20-450 Hz)")
    print("  ✓ Notch filtering (50/60 Hz power line)")
    print("  ✓ RMS envelope extraction")
    print("  ✓ TMR sensor synthesis")
    print("  ✓ sEMG channel mapping")
    print("  ✓ IMU angle derivation")
    print("  ✓ Data augmentation")
    
    print("\nExample usage:")
    print("""
from data_loader import AdvancedDataLoader
from preprocessing import SpinalBypassConverter

# Load data
loader = AdvancedDataLoader()
emg, labels = loader.load_ninapro('S1_E1_A1.mat')

# Convert to spinal bypass format
converter = SpinalBypassConverter()
sensors = converter.convert(emg, sampling_rate=200, preprocess=True)

# Now you have:
# sensors['tmr']  - (N, 8)  spinal nerve activity
# sensors['semg'] - (N, 64) muscle activity
# sensors['imu']  - (N, 3)  joint angles
    """)
