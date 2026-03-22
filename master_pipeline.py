"""
Complete end-to-end system integration

Workflow:
1. Load real EMG data
2. Preprocess to spinal bypass format
3. Extract features 
4. Train ML model
5. Generate predictions
6. Analyze performance
7. Export for Blender
"""

import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional

# Import all modules
from data_loader_UPDATED import LegEMGDataLoader
from preprocessing import SpinalBypassConverter
from feature_extraction import FeatureExtractor
from ml_models import AdvancedIntentDecoder
from analysis import PerformanceAnalyzer
from blender_export import BlenderDataExporter
from decision_layer import IntentDecisionPolicy


class SpinalBypassPipeline:
    """Complete integrated pipeline"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.data_loader = LegEMGDataLoader()
        self.preprocessor = SpinalBypassConverter()
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.analyzer = PerformanceAnalyzer(str(self.output_dir / "analysis"))
        self.blender_exporter = BlenderDataExporter(str(self.output_dir / "blender_data"))
        self.decision_policy = IntentDecisionPolicy()

        # Data
        self.raw_data = None
        self.labels = None
        self.sensors = None
        self.features = None

        # Results
        self.predictions = None
        self.confidences = None
        self.latencies = None
        self.policy_metrics = None

        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          SPINAL BYPASS SYSTEM - MASTER PIPELINE                      ║
║                                                                      ║
║    Real Data → ML Training → Analysis → Blender Visualization       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)

    def run_complete_pipeline(self,
                              data_path: str,
                              model_type: str = 'ensemble',
                              optimize_ml: bool = True,
                              n_samples: Optional[int] = None,
                              export_blender: bool = True,
                              use_decision_policy: bool = True):
        """Run complete end-to-end pipeline"""
        start_time = time.time()

        # Step 1: Load Data
        print("\n" + "="*70)
        print("STEP 1/7: LOADING DATA")
        print("="*70)
        self._load_data(data_path)

        # Limit raw data before preprocess (avoids odd-length DSP / wavelet issues)
        if n_samples is not None:
            n0 = len(self.labels) if self.labels is not None else len(self.raw_data)
            print(f"\nLimiting to {n_samples:,} samples (from {n0:,})")
            self.raw_data = self.raw_data[:n_samples]
            if self.labels is not None:
                self.labels = self.labels[:n_samples]

        # Step 2: Preprocess
        print("\n" + "="*70)
        print("STEP 2/7: PREPROCESSING")
        print("="*70)
        self._preprocess_data()

        # Step 3: Feature Extraction
        print("\n" + "="*70)
        print("STEP 3/7: FEATURE EXTRACTION")
        print("="*70)
        self._extract_features()

        # Step 4: Train Model
        print("\n" + "="*70)
        print("STEP 4/7: TRAINING ML MODEL")
        print("="*70)
        self._train_model(model_type, optimize_ml)

        # Step 5: Generate Predictions
        print("\n" + "="*70)
        print("STEP 5/7: GENERATING PREDICTIONS")
        print("="*70)
        self._generate_predictions(use_decision_policy=use_decision_policy)

        # Step 6: Analyze Performance
        print("\n" + "="*70)
        print("STEP 6/7: PERFORMANCE ANALYSIS")
        print("="*70)
        results = self._analyze_performance()

        # Step 7: Export for Blender
        if export_blender:
            print("\n" + "="*70)
            print("STEP 7/7: EXPORTING FOR BLENDER")
            print("="*70)
            self._export_for_blender()

        # Summary
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.2%}" if self.labels is not None else "  Accuracy: N/A")
        print(f"  Mean Latency: {results['mean_latency_ms']:.1f} ms")
        if self.policy_metrics:
            print(f"  P99 Latency:  {self.policy_metrics.get('p99_latency_ms', 0):.1f} ms")
            print(f"  Abstention:   {self.policy_metrics.get('abstention_rate', 0):.1%}")
        print(f"  Output Directory: {self.output_dir}")
        print("\nGenerated Files:")
        print(f"  • {self.output_dir}/trained_model.pkl" if self.labels is not None else "  • No trained model (labels missing)")
        print(f"  • {self.output_dir}/analysis/ (graphs + reports)")
        if export_blender:
            print(f"  • {self.output_dir}/blender_data/session_data.json")
        print("\n" + "="*70)

        return results

    def _load_data(self, data_path: str):
        """Load data from file (.mat, .csv, .txt) with optional label extraction"""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.suffix == '.mat':
            self.raw_data, self.labels = self.data_loader.load_ninapro(
                str(data_path), database='DB3', validate=True
            )
        elif data_path.suffix == '.csv':
            self.raw_data, self.labels = self.data_loader.load_csv(str(data_path), validate=True)
        elif data_path.suffix == '.txt':
            cleaned_lines = []
            with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.replace("ï¿½", "").replace("í", "")
                    cleaned_lines.append(line)
            arr = np.genfromtxt(cleaned_lines, delimiter=None, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[1] > 1:
                self.raw_data = arr[:, :-1]
                self.labels = arr[:, -1].astype(int)
            else:
                self.raw_data = arr
                self.labels = None
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        num_samples = len(self.labels) if self.labels is not None else self.raw_data.shape[0]
        num_features = self.raw_data.shape[1] if self.raw_data.ndim > 1 else 1
        num_classes = len(set(self.labels)) if self.labels is not None else 'N/A'
        print(f"\nDataset Statistics:")
        print(f"  Samples: {num_samples}")
        print(f"  Features per sample: {num_features}")
        print(f"  Classes: {num_classes}")

    def _preprocess_data(self):
        """Preprocess to spinal bypass format"""
        self.sensors = self.preprocessor.convert(
            self.raw_data,
            sampling_rate=self.data_loader.sampling_rate or 200,
            preprocess=True
        )
        self.sensors['labels'] = self.labels

    def _extract_features(self):
        """Extract ML features"""
        print(f"\nExtracting {self.feature_extractor.FEATURE_DIMS['total']}-dimensional features...")
        start = time.time()
        self.features = self.feature_extractor.extract_batch(
            self.sensors['tmr'],
            self.sensors['semg'],
            self.sensors['imu']
        )
        extract_time = time.time() - start
        num_samples = len(self.labels) if self.labels is not None else self.features.shape[0]
        print(f"✓ Feature extraction complete:")
        print(f"  Shape: {self.features.shape}")
        print(f"  Time: {extract_time:.1f}s ({num_samples/extract_time:.0f} samples/sec)")

    def _train_model(self, model_type: str, optimize: bool):
        """Train ML model"""
        if self.labels is None:
            print("\n⚠️  Labels missing — skipping ML training.")
            return
        self.model = AdvancedIntentDecoder(model_type=model_type)
        self.model.train(
            self.features,
            self.labels,
            test_size=0.2,
            val_size=0.1,
            optimize=optimize
        )
        self.model.save(str(self.output_dir / "trained_model.pkl"))
        if self.model.feature_importance is not None:
            feature_names = self.feature_extractor.get_feature_names()
            print(self.model.get_feature_importance_report(feature_names))

    def _generate_predictions(self, use_decision_policy: bool = True):
        """Generate predictions on full dataset"""
        num_samples = len(self.labels) if self.labels is not None else self.features.shape[0]
        print(f"\nGenerating predictions for {num_samples:,} samples...")
        if self.model is None:
            self.predictions = np.zeros(num_samples)
            self.confidences = np.zeros(num_samples)
            self.latencies = np.zeros(num_samples)
            self.policy_metrics = None
            print("⚠️  No trained model — predictions skipped")
            return
        predictions, confidences, latencies_ms = self.model.predict(self.features)
        self.latencies = np.asarray(latencies_ms).reshape(-1)
        self.confidences = confidences
        self.predictions = predictions
        self.policy_metrics = None

        if use_decision_policy and self.labels is not None:
            dr = self.decision_policy.apply(
                predictions=np.asarray(predictions),
                confidences=np.asarray(confidences),
                latencies_ms=self.latencies,
                tmr_data=self.sensors["tmr"],
                true_labels=np.asarray(self.labels),
            )
            self.predictions = dr.predictions
            self.policy_metrics = dr.policy_metrics
            print(f"✓ Decision policy: abstention {dr.policy_metrics.get('abstention_rate', 0):.1%}, "
                  f"mean TMR SNR (linear) {dr.policy_metrics.get('mean_tmr_snr_linear', 0):.2f}")

        print(f"✓ Predictions complete:")
        print(f"  Mean latency: {float(np.mean(self.latencies)):.1f} ms | "
              f"P99: {float(np.percentile(self.latencies, 99)):.1f} ms")
        print(f"  Total samples: {len(self.predictions):,}")

    def _analyze_performance(self) -> Dict:
        """Comprehensive performance analysis"""
        results = self.analyzer.analyze_complete_session(
            true_labels=self.labels,
            predictions=self.predictions,
            confidences=self.confidences,
            latencies=self.latencies,
            tmr_data=self.sensors['tmr'],
            imu_data=self.sensors['imu'],
            feature_importance=self.model.feature_importance if self.model else None,
            feature_names=self.feature_extractor.get_feature_names(),
            policy_metrics=self.policy_metrics,
        )
        return results

    def _export_for_blender(self):
        """Export data for Blender visualization"""
        output_path = self.blender_exporter.export_complete_session(
            tmr_data=self.sensors['tmr'],
            semg_data=self.sensors['semg'],
            imu_data=self.sensors['imu'],
            predictions=self.predictions,
            confidences=self.confidences,
            true_labels=self.labels,
            sampling_rate=50
        )
        print(f"\n✓ Blender data ready:")
        print(f"  Load this file in Blender: {output_path}")


# ========================
# COMMAND-LINE INTERFACE
# ========================

def main():
    parser = argparse.ArgumentParser(
        description="Spinal Bypass System - Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data', type=str, required=True, help='Path to data file (.mat, .csv, .txt)')
    parser.add_argument('--model', choices=['random_forest','xgboost','neural_net','ensemble'], default='ensemble')
    parser.add_argument('--quick', action='store_true', help='Skip ML optimization')
    parser.add_argument('--samples', type=int, default=None, help='Limit to N samples')
    parser.add_argument('--no-blender', action='store_true', help='Skip Blender export')
    parser.add_argument('--no-policy', action='store_true', help='Skip abstention / SNR gating')
    parser.add_argument('--output', type=str, default='output', help='Output directory')

    args = parser.parse_args()
    pipeline = SpinalBypassPipeline(output_dir=args.output)
    results = pipeline.run_complete_pipeline(
        data_path=args.data,
        model_type=args.model,
        optimize_ml=not args.quick,
        n_samples=args.samples,
        export_blender=not args.no_blender,
        use_decision_policy=not args.no_policy,
    )

    print("\n✓ Pipeline execution successful!")
    print(f"\nNext steps:")
    print(f"  1. Check analysis graphs: {args.output}/analysis/")
    if not args.no_blender:
        print(f"  2. Open Blender and load: {args.output}/blender_data/session_data.json")
    print(f"  3. View detailed report: {args.output}/analysis/performance_report.txt")


if __name__ == "__main__":
    main()
