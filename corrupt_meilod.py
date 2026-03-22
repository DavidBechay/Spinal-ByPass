"""
Add realistic noise to full MEILoD dataset for robustness testing

Outputs:
- corrupted_meilod_full.csv (500k+ samples with noise)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project path
sys.path.insert(0, str(Path(__file__).parent)) 

def _inject_label_noise(labels, flip_frac):
    """Flip fraction of labels randomly"""
    if flip_frac <= 0 or flip_frac >= 1:
        return labels

    np.random.seed(456)
    n = len(labels)
    flip_n = int(n * flip_frac)
    flip_idx = np.random.choice(n, flip_n, replace=False)

    unique_labels = np.unique(labels)
    for i in flip_idx:
        current = labels[i]
        choices = [c for c in unique_labels if c != current]
        if choices:
            labels[i] = np.random.choice(choices)

    return labels

def _inject_gaussian_noise(data, mean=0.0, std=1.5):
    """Add Gaussian noise to data"""
    np.random.seed(42)
    noise = np.random.normal(mean, std, size=data.shape)
    return data + noise

def _inject_crosstalk_spikes(data, events=3):
    """Add crosstalk spikes to random EMG channels"""
    np.random.seed(789)
    N, C = data.shape
    for _ in range(events):
        start = np.random.randint(0, max(1, N - 50))
        width = np.random.randint(10, 70)
        end = min(N, start + width)
        amplitude = np.random.uniform(2.0, 4.0)
        # Add to random channels
        channels = np.random.choice(C, size=np.random.randint(1, C//2), replace=False)
        data[start:end, channels] += amplitude + np.random.normal(0, 0.3, size=(end-start, len(channels)))

    return data

def main():
    print("🔧 CORRUPTING MEILoD DATASET")
    print("="*50)

    # Load full dataset
    data_path = Path("MEILoD_v1.1_merged.csv")
    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        print("Download from: https://data.mendeley.com/datasets/ydz48cby4t/1")
        sys.exit(1)

    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} samples")

    # Identify columns
    emg_cols = [col for col in df.columns if '(mV)' in col or any(muscle in col for muscle in ['Rectus_Femoris', 'Vastus', 'Semitendinosus'])]
    imu_cols = [col for col in df.columns if 'ACC' in col.upper() or 'GYRO' in col.upper()]
    label_cols = [col for col in df.columns if any(x in col.lower() for x in ['activity', 'label', 'class'])]

    print(f"EMG channels: {len(emg_cols)}")
    print(f"IMU channels: {len(imu_cols)}")
    print(f"Label column: {label_cols[0] if label_cols else 'None'}")

    # Apply label corruption
    if label_cols:
        print("\n⚠️ Injecting label flip noise (10%)...")
        df[label_cols[0]] = _inject_label_noise(df[label_cols[0]].values, 0.10)

    # Apply Gaussian noise to EMG/IMU
    all_sensor_cols = emg_cols + imu_cols
    if all_sensor_cols:
        print("⚠️ Injecting Gaussian noise to sensor data...")
        sensor_data = df[all_sensor_cols].values
        sensor_data = _inject_gaussian_noise(sensor_data, mean=0.0, std=1.5)
        df[all_sensor_cols] = sensor_data

    # Apply crosstalk spikes to EMG
    if emg_cols:
        print("⚠️ Injecting crosstalk spikes to EMG...")
        emg_data = df[emg_cols].values
        emg_data = _inject_crosstalk_spikes(emg_data, events=3)
        df[emg_cols] = emg_data

    # Save corrupted dataset
    output_path = Path("corrupted_meilod_full.csv")
    print(f"\n💾 Saving corrupted dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df):,} samples")

    print("\n🎯 Corrupted dataset ready for analysis!")
    print("Run: python 00_quick_start_meilod.py --data corrupted_meilod_full.csv --samples 50000 --quick")

if __name__ == "__main__":
    main()
