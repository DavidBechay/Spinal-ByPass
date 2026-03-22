"""
1. IEEE DataPort 2023 (RECOMMENDED - Newest)
2. Zenodo 2022
3. PhysioNet EMG Gait
4. Synthetic (fallback)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class NewestDataLoader:
    """
    Load the newest (2023-2024) leg EMG datasets
    """
    
    def __init__(self):
        self.data = None
        self.labels = None
        self.sampling_rate = None
        self.dataset_name = None
        self.muscle_names = []
    
    def load_ieee_dataport_2023(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load IEEE DataPort 2023 Gait EMG Dataset (NEWEST!)
        
        Download from: 
        https://ieee-dataport.org/open-access/surface-electromyography-dataset-gait-analysis
        
        This is the NEWEST publicly available leg EMG dataset (2023)
        
        Args:
            filepath: Path to .mat file from IEEE DataPort
        
        Returns:
            data: (N, channels) EMG data
            labels: (N,) movement labels
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required. Install: pip install scipy")
        
        print(f"\n{'='*70}")
        print("LOADING IEEE DATAPORT 2023 GAIT EMG (NEWEST DATASET)")
        print(f"{'='*70}")
        print(f"File: {filepath}")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"File not found: {filepath}\n\n"
                f"Download from:\n"
                f"https://ieee-dataport.org/open-access/surface-electromyography-dataset-gait-analysis\n\n"
                f"Steps:\n"
                f"1. Visit URL above\n"
                f"2. Create FREE IEEE account (takes 1 minute)\n"
                f"3. Click 'Download'\n"
                f"4. Extract .mat files\n"
                f"5. Use any subject file (e.g., Subject01.mat)\n"
            )
        
        # Load .mat file
        mat_data = loadmat(str(filepath))
        
        # IEEE DataPort structure (may vary by specific file)
        # Common fields: 'EMG', 'emg', 'data', 'signals'
        
        possible_data_keys = ['EMG', 'emg', 'data', 'signals', 'emg_data']
        possible_label_keys = ['labels', 'phase', 'gait_phase', 'activity']
        
        emg = None
        labels = None
        
        # Find EMG data
        for key in possible_data_keys:
            if key in mat_data:
                emg = mat_data[key]
                print(f"  Found EMG data in field: '{key}'")
                break
        
        if emg is None:
            print(f"\n⚠ Available fields in .mat file:")
            for key in mat_data.keys():
                if not key.startswith('__'):
                    print(f"  - {key}: {type(mat_data[key])}")
            raise ValueError(
                "Could not find EMG data. Please check the .mat file structure.\n"
                "Try: data_loader.load_csv() if this is a CSV file instead."
            )
        
        # Find labels
        for key in possible_label_keys:
            if key in mat_data:
                labels = mat_data[key].flatten()
                print(f"  Found labels in field: '{key}'")
                break
        
        # If no labels, generate from gait cycle
        if labels is None:
            print("  No labels found - generating from gait phases...")
            # Simple gait phase detection
            # Assume channel 0 is tibialis anterior (peaks at heel strike)
            if emg.shape[1] >= 1:
                sr = self.sampling_rate or 1000
                self.sampling_rate = sr
                tibialis = emg[:, 0]
                # Detect peaks (heel strikes)
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(tibialis, distance=int(1.0 * sr))
                
                # Label between peaks as gait phases
                labels = np.array(['stance'] * len(emg))
                for i in range(len(peaks)-1):
                    cycle_start = peaks[i]
                    cycle_end = peaks[i+1]
                    cycle_len = cycle_end - cycle_start
                    
                    # 60% stance, 40% swing
                    swing_start = cycle_start + int(0.6 * cycle_len)
                    labels[swing_start:cycle_end] = 'swing'
        
        print(f"\n✓ Loaded successfully:")
        print(f"  Samples: {emg.shape[0]:,}")
        print(f"  Channels: {emg.shape[1]}")
        
        # IEEE DataPort typically has 8-10 channels
        self.muscle_names = [
            'tibialis_anterior_L',
            'gastrocnemius_L',
            'vastus_lateralis_L',
            'biceps_femoris_L',
            'rectus_femoris_L',
            'tibialis_anterior_R',
            'gastrocnemius_R',
            'vastus_lateralis_R',
        ][:emg.shape[1]]  # Limit to actual channels
        
        self.data = emg
        self.labels = labels if labels is not None else np.array(['walking'] * len(emg))
        self.sampling_rate = 1000  # IEEE DataPort typically 1000 Hz
        self.dataset_name = "IEEE DataPort 2023"
        
        print(f"  Sampling Rate: {self.sampling_rate} Hz")
        print(f"  Duration: {emg.shape[0]/self.sampling_rate:.1f} seconds")
        print(f"  Muscles: {', '.join(self.muscle_names)}")
        
        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\n  Movement Phases:")
            for label, count in zip(unique, counts):
                print(f"    {label}: {count:,} samples ({count/len(labels)*100:.1f}%)")
        
        return emg, self.labels
    
    def load_zenodo_2022(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Zenodo 2022 Lower Limb EMG Dataset
        
        Download from: https://zenodo.org/record/6457662
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            data: (N, channels) EMG
            labels: (N,) activity labels
        """
        print(f"\n{'='*70}")
        print("LOADING ZENODO 2022 LOWER LIMB EMG")
        print(f"{'='*70}")
        
        if not PANDAS_AVAILABLE:
            # Fallback to numpy
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            
            # Assume last column is labels
            if data.shape[1] > 8:
                labels = data[:, -1].astype(int)
                emg = data[:, :-1]
            else:
                emg = data
                labels = np.array(['walking'] * len(data))
        else:
            df = pd.read_csv(filepath)
            
            # Common column names
            label_cols = ['activity', 'label', 'phase', 'movement']
            
            labels = None
            for col in label_cols:
                if col in df.columns:
                    labels = df[col].values
                    emg = df.drop(columns=[col]).values
                    break
            
            if labels is None:
                emg = df.values
                labels = np.array(['walking'] * len(df))
        
        # Map numeric labels to names
        if labels.dtype in [np.int32, np.int64, np.float64]:
            label_map = {
                0: 'rest',
                1: 'walking',
                2: 'running',
                3: 'stairs_up',
                4: 'stairs_down',
                5: 'sitting',
                6: 'standing'
            }
            labels = np.array([label_map.get(int(l), 'walking') for l in labels])
        
        self.data = emg
        self.labels = labels
        self.sampling_rate = 1000
        self.dataset_name = "Zenodo 2022"
        
        print(f"✓ Loaded: {emg.shape[0]:,} samples × {emg.shape[1]} channels")
        
        return emg, labels
    
    def auto_detect_and_load(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatically detect file format and load
        
        Args:
            filepath: Path to data file
        
        Returns:
            data, labels
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.mat':
            return self.load_ieee_dataport_2023(str(filepath))
        elif filepath.suffix == '.csv':
            return self.load_zenodo_2022(str(filepath))
        elif filepath.suffix == '.txt':
            return self.load_zenodo_2022(str(filepath))
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")

    def generate_synthetic_gait_emg(
        self,
        duration_seconds: float = 60.0,
        sampling_rate: int = 200,
        n_cycles: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Physiological synthetic gait EMG (Perry & Burnfield–style)."""
        return LegEMGDataLoader().generate_synthetic_gait_emg(
            duration_seconds=duration_seconds,
            sampling_rate=sampling_rate,
            n_cycles=n_cycles,
        )


class LegEMGDataLoader:
    """
    Lower-limb EMG: PhysioNet, synthetic gait, NinaPro (.mat), and generic CSV.
    """

    def __init__(self):
        self.data = None
        self.labels = None
        self.sampling_rate = None
        self.dataset_name = None
        self.muscle_names = []

    def load_ninapro(
        self,
        filepath: str,
        database: str = "DB3",
        validate: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load NinaPro DB3 (or similar) .mat with 'emg' and optional 'restimulus'/'stimulus'."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required. Install: pip install scipy")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        mat = loadmat(str(filepath))
        emg = mat.get("emg")
        if emg is None:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise ValueError(
                f"No 'emg' key in {filepath.name}. Available keys: {keys}"
            )

        restim = mat.get("restimulus")
        if restim is None:
            restim = mat.get("stimulus")
        if restim is not None:
            y = np.asarray(restim).flatten()
            n = min(len(y), emg.shape[0])
            labels = np.zeros(emg.shape[0], dtype=np.int32)
            labels[:n] = y[:n].astype(np.int32)
            # NinaPro often uses 1–12; fold to 0-based classes
            labels = np.where(labels > 0, labels - 1, labels)
            labels = np.clip(labels, 0, 11)
        else:
            labels = np.zeros(emg.shape[0], dtype=np.int32)

        self.data = emg
        self.labels = labels
        self.sampling_rate = 200
        self.dataset_name = f"NinaPro {database}"
        self.muscle_names = [f"ch{i}" for i in range(emg.shape[1])]

        print(f"\n[OK] NinaPro loaded: {emg.shape[0]:,} x {emg.shape[1]} channels, {len(np.unique(labels))} classes")
        return emg, labels

    def load_csv(self, filepath: str, validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """CSV with optional label column named activity/label/phase/movement (last col fallback)."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)

        if PANDAS_AVAILABLE:
            df = pd.read_csv(filepath)
            label_cols = ["activity", "label", "phase", "movement", "class", "y"]
            labels = None
            for col in label_cols:
                if col in df.columns:
                    labels = df[col].values
                    emg = df.drop(columns=[col]).values.astype(float)
                    break
            if labels is None:
                if df.shape[1] > 1:
                    emg = df.iloc[:, :-1].values.astype(float)
                    labels = df.iloc[:, -1].values
                else:
                    emg = df.values.astype(float)
                    labels = np.zeros(len(emg), dtype=np.int32)
        else:
            data = np.loadtxt(filepath, delimiter=",", skiprows=1)
            if data.shape[1] > 1:
                emg = data[:, :-1]
                labels = data[:, -1]
            else:
                emg = data
                labels = np.zeros(len(emg), dtype=np.int32)

        if labels.dtype in (np.float64, np.float32):
            labels = labels.astype(np.int32)

        self.data = emg
        self.labels = labels
        self.sampling_rate = 1000
        self.dataset_name = "CSV"
        self.muscle_names = [f"ch{i}" for i in range(emg.shape[1])]
        print(f"\n[OK] CSV loaded: {emg.shape[0]:,} x {emg.shape[1]} channels")
        return emg, labels

    def load_physionet_emg(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """PhysioNet EMG gait database (tab/space-separated)."""
        print(f"\n{'='*70}")
        print("LOADING PHYSIONET LEG EMG DATA")
        print(f"{'='*70}")
        print(f"File: {filepath}")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"File not found: {filepath}\n\n"
                f"Download from: https://physionet.org/content/emgdb/1.0.0/\n"
            )

        try:
            data = np.loadtxt(filepath)
        except Exception as e:
            if PANDAS_AVAILABLE:
                df = pd.read_csv(filepath, sep=r"\s+", header=None)
                data = df.values
            else:
                raise e

        print(f"\n[OK] Loaded successfully:")
        print(f"  Samples: {data.shape[0]:,}")
        print(f"  Channels: {data.shape[1]}")

        self.muscle_names = [
            "tibialis_anterior_L",
            "gastrocnemius_L",
            "vastus_lateralis_L",
            "biceps_femoris_L",
        ]
        if data.shape[1] >= 8:
            self.muscle_names.extend(
                [
                    "tibialis_anterior_R",
                    "gastrocnemius_R",
                    "vastus_lateralis_R",
                    "biceps_femoris_R",
                ]
            )

        tibialis = data[:, 0]
        threshold = np.mean(tibialis) + 0.5 * np.std(tibialis)
        labels = np.where(tibialis > threshold, "swing_phase", "stance_phase")
        swing_mask = labels == "swing_phase"
        if data.shape[1] >= 3:
            vastus = data[:, 2]
            knee_ext_mask = swing_mask & (vastus > np.median(vastus))
            labels[knee_ext_mask] = "knee_extend"

        self.data = data
        self.labels = labels
        self.sampling_rate = 1000
        self.dataset_name = "PhysioNet Leg EMG"

        print(f"  Sampling Rate: {self.sampling_rate} Hz")
        return data, labels

    def generate_synthetic_gait_emg(
        self,
        duration_seconds: float = 60.0,
        sampling_rate: int = 200,
        n_cycles: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthetic bilateral gait EMG (published gait timing patterns)."""
        print(f"\n{'='*70}")
        print("GENERATING SYNTHETIC GAIT EMG")
        print(f"{'='*70}")

        N = int(duration_seconds * sampling_rate)
        emg = np.zeros((N, 8))
        labels = np.array(["stance"] * N)

        cycle_duration = duration_seconds / n_cycles
        samples_per_cycle = int(cycle_duration * sampling_rate)

        muscle_patterns = {
            "tibialis_anterior": {"peak_times": [0.0, 0.60], "peak_values": [0.8, 1.0], "width": 0.15},
            "gastrocnemius": {"peak_times": [0.45], "peak_values": [1.0], "width": 0.20},
            "vastus_lateralis": {"peak_times": [0.10, 0.65], "peak_values": [0.9, 0.7], "width": 0.15},
            "biceps_femoris": {"peak_times": [0.50, 0.85], "peak_values": [0.8, 0.6], "width": 0.12},
        }

        for cycle in range(n_cycles):
            start_idx = cycle * samples_per_cycle
            end_idx = min((cycle + 1) * samples_per_cycle, N)
            cycle_samples = end_idx - start_idx
            t = np.linspace(0, 1, cycle_samples)

            for muscle_idx, (muscle_name, pattern) in enumerate(muscle_patterns.items()):
                activation = np.zeros(cycle_samples)
                for peak_time, peak_value in zip(pattern["peak_times"], pattern["peak_values"]):
                    activation += peak_value * np.exp(
                        -((t - peak_time) ** 2) / (2 * pattern["width"] ** 2)
                    )
                activation += np.random.normal(0, 0.05, cycle_samples)
                activation = np.clip(activation, 0, 1) * 1.5 + 0.1
                emg[start_idx:end_idx, muscle_idx] = activation

                t_shifted = (t + 0.5) % 1.0
                activation_R = np.zeros(cycle_samples)
                for peak_time, peak_value in zip(pattern["peak_times"], pattern["peak_values"]):
                    activation_R += peak_value * np.exp(
                        -((t_shifted - peak_time) ** 2) / (2 * pattern["width"] ** 2)
                    )
                activation_R += np.random.normal(0, 0.05, cycle_samples)
                activation_R = np.clip(activation_R, 0, 1) * 1.5 + 0.1
                emg[start_idx:end_idx, muscle_idx + 4] = activation_R

            stance_end = int(0.60 * cycle_samples)
            labels[start_idx:start_idx + stance_end] = "stance_phase"
            labels[start_idx + stance_end:end_idx] = "swing_phase"
            loading_end = int(0.10 * cycle_samples)
            midstance_end = int(0.30 * cycle_samples)
            labels[start_idx:start_idx + loading_end] = "heel_strike"
            labels[start_idx + loading_end : start_idx + midstance_end] = "stance_phase"
            labels[start_idx + midstance_end : start_idx + stance_end] = "push_off"
            swing_mid = start_idx + stance_end + int(0.13 * cycle_samples)
            labels[start_idx + stance_end:swing_mid] = "knee_extend"
            labels[swing_mid:end_idx] = "swing_phase"

        self.data = emg
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.dataset_name = "Synthetic Gait EMG"
        self.muscle_names = [
            "tibialis_anterior_L",
            "gastrocnemius_L",
            "vastus_lateralis_L",
            "biceps_femoris_L",
            "tibialis_anterior_R",
            "gastrocnemius_R",
            "vastus_lateralis_R",
            "biceps_femoris_R",
        ]

        print(f"\n[OK] Synthetic: {N:,} samples x 8 channels")
        return emg, labels


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              NEWEST LEG EMG DATA SOURCES (2023-2024)                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

OPTION 1: IEEE DataPort 2023 (RECOMMENDED - Newest!)
─────────────────────────────────────────────────────
URL: https://ieee-dataport.org/open-access/surface-electromyography-dataset-gait-analysis
Year: 2023
Subjects: 20 healthy adults
Muscles: Tibialis, gastrocnemius, vastus, biceps femoris, rectus femoris
Activities: Normal walking
Format: .mat files
Free: Yes (requires free IEEE account)

Download Steps:
1. Visit URL above
2. Create FREE IEEE DataPort account (1 minute)
3. Click "Download" button
4. Extract files
5. Use: loader.load_ieee_dataport_2023('Subject01.mat')


OPTION 2: Zenodo 2022
─────────────────────────────────────────────────────
URL: https://zenodo.org/record/6457662
Year: 2022
Activities: Walking, running, stairs
Format: CSV
Free: Yes (direct download, no account)

Download Steps:
1. Visit URL
2. Click "Download"
3. Use: loader.load_zenodo_2022('data.csv')


OPTION 3: OpenNeuro 2024 (Research-grade)
─────────────────────────────────────────────────────
URL: https://openneuro.org/
Search: "gait" or "lower extremity"
Year: 2024 datasets available
Format: BIDS standard
Quality: Highest (research-grade)

QUICK START:
───────────────────────────────────────────────────── 
from newest_data_loader import NewestDataLoader

loader = NewestDataLoader()

# Auto-detect format
emg, labels = loader.auto_detect_and_load('your_file.mat')

# Or specify dataset
emg, labels = loader.load_ieee_dataport_2023('Subject01.mat')
    """)
