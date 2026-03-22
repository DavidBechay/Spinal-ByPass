"""
ADVANCED FEATURE EXTRACTION
Comprehensive feature set for movement intent decoding
 
Feature Categories:
1. TMR Features (24)
   - Raw values
   - Differentials
   - Statistics
   - Temporal patterns

2. sEMG Features (32)
   - RMS per muscle group
   - Mean Absolute Value
   - Waveform Length
   - Zero Crossings
   - Slope Sign Changes

3. IMU Features (16)
   - Joint angles
   - Angular velocities (estimated)
   - Joint coupling
   - Kinematic chains

4. Cross-Modal Features (8)
   - TMR-sEMG correlation
   - sEMG-IMU correlation
   - Asymmetry indices

Total: 80 features
"""

import numpy as np
from typing import Tuple, List
import warnings


class FeatureExtractor:
    """
    Advanced feature extraction for movement intent decoding
    """
    
    # Feature dimension specs
    FEATURE_DIMS = {
        'tmr': 32,  # Increased from 24
        'semg': 32,
        'imu': 16,
        'cross_modal': 8,
        'total': 88  # Increased from 80
    }
    
    @staticmethod
    def extract_tmr_features(tmr: np.ndarray) -> np.ndarray:
        """
        Extract TMR sensor features (32 features) - Enhanced for noisy data
        
        Args:
            tmr: (8,) TMR sensor readings
        
        Returns:
            (32,) feature vector
        """
        features = np.zeros(32)
        
        # Raw readings (8)
        features[0:8] = tmr
        
        # Differential signals (4)
        features[8] = tmr[0] - tmr[1]   # L1 left-right
        features[9] = tmr[2] - tmr[3]   # L2 left-right
        features[10] = tmr[4] - tmr[5]  # L3 left-right
        features[11] = tmr[6] - tmr[7]  # L4 left-right
        
        # Cross-level differentials (4)
        features[12] = tmr[0:4].sum() - tmr[4:8].sum()  # L1/L2 vs L3/L4
        features[13] = tmr[[0,2,4,6]].sum() - tmr[[1,3,5,7]].sum()  # Left vs Right
        features[14] = np.mean(tmr[0:4]) - np.mean(tmr[4:8])  # Upper vs Lower mean
        features[15] = np.std(tmr[0:4]) - np.std(tmr[4:8])    # Upper vs Lower variance
        
        # Statistics (8) - Robust versions
        features[16] = np.mean(tmr)
        features[17] = np.std(tmr)
        features[18] = np.max(tmr)
        features[19] = np.min(tmr)
        features[20] = np.median(tmr)  # More robust than mean
        features[21] = np.ptp(tmr)  # Peak-to-peak (range)
        features[22] = np.percentile(tmr, 75) - np.percentile(tmr, 25)  # IQR
        features[23] = np.mean(np.abs(tmr - np.median(tmr)))  # MAD
        
        # Asymmetry indices (6)
        left = tmr[[0,2,4,6]]
        right = tmr[[1,3,5,7]]
        features[24] = np.mean(np.abs(left - right))  # Mean asymmetry
        features[25] = np.max(np.abs(left - right))   # Max asymmetry
        features[26] = np.std(left) - np.std(right)   # Variance asymmetry
        features[27] = np.median(left) - np.median(right)  # Median asymmetry
        
        # Relative activations (6)
        total = tmr.sum() + 1e-10
        features[28] = tmr[0:4].sum() / total  # L1/L2 proportion
        features[29] = tmr[4:8].sum() / total  # L3/L4 proportion
        features[30] = left.sum() / total      # Left proportion
        features[31] = right.sum() / total     # Right proportion
        
        return features
    
    @staticmethod
    def extract_semg_features(semg: np.ndarray) -> np.ndarray:
        """
        Extract sEMG features (32 features)
        
        Args:
            semg: (64,) sEMG channels
        
        Returns:
            (32,) feature vector
        """
        features = np.zeros(32)
        
        # Define muscle groups (16 channels each)
        muscle_groups = {
            'iliopsoas': semg[0:16],
            'quadriceps': semg[16:32],
            'hamstrings': semg[32:48],
            'tibialis': semg[48:64],
        }
        
        idx = 0
        
        for muscle_name, muscle_data in muscle_groups.items():
            # RMS (Root Mean Square)
            features[idx] = np.sqrt(np.mean(muscle_data ** 2))
            idx += 1
            
            # MAV (Mean Absolute Value)
            features[idx] = np.mean(np.abs(muscle_data))
            idx += 1
            
            # WL (Waveform Length)
            features[idx] = np.sum(np.abs(np.diff(muscle_data)))
            idx += 1
            
            # ZC (Zero Crossings)
            features[idx] = np.sum((muscle_data[:-1] * muscle_data[1:]) < 0)
            idx += 1
            
            # SSC (Slope Sign Changes)
            diff1 = np.diff(muscle_data)
            features[idx] = np.sum((diff1[:-1] * diff1[1:]) < 0)
            idx += 1
            
            # Variance
            features[idx] = np.var(muscle_data)
            idx += 1
            
            # Max value
            features[idx] = np.max(np.abs(muscle_data))
            idx += 1
            
            # 75th percentile
            features[idx] = np.percentile(np.abs(muscle_data), 75)
            idx += 1
        
        return features
    
    @staticmethod
    def extract_imu_features(imu: np.ndarray, 
                            prev_imu: np.ndarray = None,
                            dt: float = 0.02) -> np.ndarray:
        """
        Extract IMU features (16 features)
        
        Args:
            imu: (3,) current joint angles [hip, knee, ankle]
            prev_imu: (3,) previous joint angles (for velocity)
            dt: Time step (seconds)
        
        Returns:
            (16,) feature vector
        """
        features = np.zeros(16)
        
        # Raw angles (3)
        features[0] = imu[0]  # Hip
        features[1] = imu[1]  # Knee
        features[2] = imu[2]  # Ankle
        
        # Angular velocities (3) - estimated from differences
        if prev_imu is not None:
            features[3] = (imu[0] - prev_imu[0]) / dt  # Hip velocity
            features[4] = (imu[1] - prev_imu[1]) / dt  # Knee velocity
            features[5] = (imu[2] - prev_imu[2]) / dt  # Ankle velocity
        else:
            features[3:6] = 0
        
        # Absolute angles (3)
        features[6] = np.abs(imu[0])
        features[7] = np.abs(imu[1])
        features[8] = np.abs(imu[2])
        
        # Joint coupling (3)
        features[9] = imu[0] + imu[1]    # Hip-knee coupling
        features[10] = imu[1] + imu[2]   # Knee-ankle coupling
        features[11] = imu[0] + imu[2]   # Hip-ankle coupling
        
        # Joint products (2)
        features[12] = imu[0] * imu[1]   # Hip × Knee
        features[13] = imu[1] * imu[2]   # Knee × Ankle
        
        # Total flexion magnitude (1)
        features[14] = np.sqrt(imu[0]**2 + imu[1]**2 + imu[2]**2)
        
        # Direction agreement (1)
        features[15] = np.sign(imu[0]) * np.sign(imu[1]) * np.sign(imu[2])
        
        return features
    
    @staticmethod
    def extract_cross_modal_features(tmr: np.ndarray,
                                    semg: np.ndarray,
                                    imu: np.ndarray) -> np.ndarray:
        """
        Extract cross-modal correlation features (8 features)
        
        Args:
            tmr: (8,) TMR sensors
            semg: (64,) sEMG channels
            imu: (3,) joint angles
        
        Returns:
            (8,) feature vector
        """
        features = np.zeros(8)
        
        # TMR-sEMG correlation (2)
        # Correlate L3/L4 TMR with quadriceps sEMG
        tmr_quad = tmr[4:8].mean()
        semg_quad = np.sqrt(np.mean(semg[16:32] ** 2))
        features[0] = tmr_quad * semg_quad
        
        # Correlate L1/L2 TMR with iliopsoas sEMG
        tmr_hip = tmr[0:4].mean()
        semg_hip = np.sqrt(np.mean(semg[0:16] ** 2))
        features[1] = tmr_hip * semg_hip
        
        # sEMG-IMU correlation (3)
        # Quadriceps activity × knee angle
        features[2] = semg_quad * np.abs(imu[1])
        
        # Iliopsoas activity × hip angle
        features[3] = semg_hip * np.abs(imu[0])
        
        # Tibialis activity × ankle angle
        semg_tib = np.sqrt(np.mean(semg[48:64] ** 2))
        features[4] = semg_tib * np.abs(imu[2])
        
        # Asymmetry correlation (2)
        tmr_asym = np.abs(tmr[[0,2,4,6]].mean() - tmr[[1,3,5,7]].mean())
        semg_asym = np.abs(semg[0:32].mean() - semg[32:64].mean())
        features[5] = tmr_asym
        features[6] = semg_asym
        
        # Overall activation level (1)
        features[7] = tmr.mean() * semg_quad * (np.abs(imu).sum() / 3)
        
        return features
    
    @staticmethod
    def extract_complete(tmr: np.ndarray,
                        semg: np.ndarray,
                        imu: np.ndarray,
                        prev_imu: np.ndarray = None) -> np.ndarray:
        """
        Extract complete 80-dimensional feature vector
        
        Args:
            tmr: (8,) TMR sensors
            semg: (64,) sEMG channels
            imu: (3,) joint angles
            prev_imu: (3,) previous joint angles
        
        Returns:
            (80,) complete feature vector
        """
        features = np.zeros(88)  # Updated from 80 to 88
        
        # TMR features (0:32)
        features[0:32] = FeatureExtractor.extract_tmr_features(tmr)
        
        # sEMG features (32:64)
        features[32:64] = FeatureExtractor.extract_semg_features(semg)
        
        # IMU features (64:80)
        features[64:80] = FeatureExtractor.extract_imu_features(imu, prev_imu)
        
        # Cross-modal features (80:88)
        features[80:88] = FeatureExtractor.extract_cross_modal_features(tmr, semg, imu)
        
        return features
    
    @staticmethod
    def extract_batch(tmr_batch: np.ndarray,
                     semg_batch: np.ndarray,
                     imu_batch: np.ndarray) -> np.ndarray:
        """
        Extract features for batch of samples
        
        Args:
            tmr_batch: (N, 8)
            semg_batch: (N, 64)
            imu_batch: (N, 3)
        
        Returns:
            (N, 80) feature matrix
        """
        N = tmr_batch.shape[0]
        features = np.zeros((N, 88))  # Updated from 80 to 88
        
        for i in range(N):
            # Use previous sample for velocity estimation
            prev_imu = imu_batch[i-1] if i > 0 else None
            
            features[i] = FeatureExtractor.extract_complete(
                tmr_batch[i],
                semg_batch[i],
                imu_batch[i],
                prev_imu
            )
        
        return features
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get human-readable feature names"""
        names = []
        
        # TMR (32)
        for i in range(8):
            names.append(f"TMR_{i}")
        for level in ['L1', 'L2', 'L3', 'L4']:
            names.append(f"TMR_diff_{level}")
        names.extend(['TMR_L12_L34_diff', 'TMR_left_right_diff', 'TMR_upper_lower_mean_diff', 'TMR_upper_lower_var_diff'])
        names.extend(['TMR_mean', 'TMR_std', 'TMR_max', 'TMR_min', 
                     'TMR_median', 'TMR_range', 'TMR_iqr', 'TMR_mad'])
        names.extend(['TMR_asym_mean', 'TMR_asym_max', 'TMR_asym_var', 'TMR_asym_median'])
        names.extend(['TMR_L12_prop', 'TMR_L34_prop', 'TMR_left_prop', 'TMR_right_prop'])
        
        # sEMG (32)
        for muscle in ['iliopsoas', 'quadriceps', 'hamstrings', 'tibialis']:
            names.extend([
                f'{muscle}_RMS',
                f'{muscle}_MAV',
                f'{muscle}_WL',
                f'{muscle}_ZC',
                f'{muscle}_SSC',
                f'{muscle}_var',
                f'{muscle}_max',
                f'{muscle}_p75'
            ])
        
        # IMU (16)
        names.extend(['hip_angle', 'knee_angle', 'ankle_angle'])
        names.extend(['hip_vel', 'knee_vel', 'ankle_vel'])
        names.extend(['hip_abs', 'knee_abs', 'ankle_abs'])
        names.extend(['hip_knee_coupling', 'knee_ankle_coupling', 'hip_ankle_coupling'])
        names.extend(['hip_knee_prod', 'knee_ankle_prod'])
        names.extend(['total_flexion', 'direction_agreement'])
        
        # Cross-modal (8)
        names.extend([
            'TMR_quad_sEMG_quad',
            'TMR_hip_sEMG_hip',
            'sEMG_quad_knee_angle',
            'sEMG_hip_hip_angle',
            'sEMG_tib_ankle_angle',
            'TMR_asymmetry',
            'sEMG_asymmetry',
            'overall_activation'
        ])
        
        return names


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              ADVANCED FEATURE EXTRACTION                             ║
║                                                                      ║
║           80-Dimensional Comprehensive Feature Set                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nFeature Categories:")
    print(f"  TMR Features:         {FeatureExtractor.FEATURE_DIMS['tmr']}")
    print(f"  sEMG Features:        {FeatureExtractor.FEATURE_DIMS['semg']}")
    print(f"  IMU Features:         {FeatureExtractor.FEATURE_DIMS['imu']}")
    print(f"  Cross-Modal Features: {FeatureExtractor.FEATURE_DIMS['cross_modal']}")
    print(f"  {'─'*40}")
    print(f"  TOTAL:                {FeatureExtractor.FEATURE_DIMS['total']}")
    
    print("\nExample usage:")
    print("""
from feature_extraction import FeatureExtractor

# Single sample
tmr = np.random.randn(8)
semg = np.random.randn(64)
imu = np.random.randn(3)

features = FeatureExtractor.extract_complete(tmr, semg, imu)
print(f"Features shape: {features.shape}")  # (80,)

# Batch processing
tmr_batch = np.random.randn(1000, 8)
semg_batch = np.random.randn(1000, 64)
imu_batch = np.random.randn(1000, 3)

features_batch = FeatureExtractor.extract_batch(
    tmr_batch, semg_batch, imu_batch
)
print(f"Batch shape: {features_batch.shape}")  # (1000, 80)

# Get feature names
names = FeatureExtractor.get_feature_names()
print(f"Feature names: {names[:5]}...")
    """)
