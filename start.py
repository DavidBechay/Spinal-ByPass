import numpy as np 

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                    QUICK START - NO DOWNLOAD                         ║
║                                                                      ║
║         Synthetic Gait EMG → Complete Pipeline → Blender             ║
║                                                                      ║ 
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "="*70)
print("STEP 1/7: GENERATING SYNTHETIC GAIT EMG")
print("="*70)
print("\nThis generates realistic leg muscle EMG based on:")
print("  • Perry & Burnfield (2010) - Gait Analysis")
print("  • Winter (2009) - Biomechanics of Human Movement")
print("\nNo download needed - physiologically accurate!")

# Import updated loader
from data_loader_UPDATED import NewestDataLoader

# Generate synthetic data
loader = NewestDataLoader()
emg, labels = loader.generate_synthetic_gait_emg(
    duration_seconds=60.0,  # 1 minute of gait
    sampling_rate=200,      # 200 Hz (standard for EMG)
    n_cycles=20             # 20 gait cycles (normal walking)
)

print("\n✓ Synthetic data ready!")
print(f"  Muscles: Tibialis, Gastrocnemius, Vastus, Biceps Femoris")
print(f"  Bilateral: Left + Right legs")
print(f"  Total: {emg.shape[0]:,} samples × {emg.shape[1]} channels")

# Now run the pipeline
print("\n" + "="*70)
print("STEP 2/7: PREPROCESSING")
print("="*70)

from preprocessing import SpinalBypassConverter

converter = SpinalBypassConverter()
sensors = converter.convert(emg, sampling_rate=200, preprocess=True)
sensors['labels'] = labels

print("\n" + "="*70)
print("STEP 3/7: FEATURE EXTRACTION")
print("="*70)

from feature_extraction import FeatureExtractor

features = FeatureExtractor.extract_batch(
    sensors['tmr'],
    sensors['semg'],
    sensors['imu']
)

print(f"\n✓ Extracted {features.shape[1]} features per sample")

print("\n" + "="*70)
print("STEP 4/7: TRAINING ML MODEL")
print("="*70)

from ml_models import AdvancedIntentDecoder

decoder = AdvancedIntentDecoder(model_type='random_forest')
accuracy = decoder.train(
    features,
    labels,
    test_size=0.2,
    optimize=False  # Quick mode
)

# Save model
decoder.save('output/trained_model.pkl')

print("\n" + "="*70)
print("STEP 5/7: GENERATING PREDICTIONS")
print("="*70)

predictions, confidences, latencies_ms = decoder.predict(features)
latencies = np.asarray(latencies_ms).reshape(-1)

print(f"\n✓ Predictions complete")
print(f"  Mean latency: {float(np.mean(latencies)):.1f} ms | P99: {float(np.percentile(latencies, 99)):.1f} ms")

print("\n" + "="*70)
print("STEP 6/7: PERFORMANCE ANALYSIS")
print("="*70)

from analysis import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(output_dir='output/analysis')

results = analyzer.analyze_complete_session(
    true_labels=labels,
    predictions=predictions,
    confidences=confidences,
    latencies=latencies,
    tmr_data=sensors['tmr'],
    imu_data=sensors['imu'],
    feature_importance=decoder.feature_importance,
    feature_names=FeatureExtractor.get_feature_names()
)

print("\n" + "="*70)
print("STEP 7/7: EXPORTING FOR BLENDER")
print("="*70)

from blender_export import BlenderDataExporter

exporter = BlenderDataExporter(output_dir='output/blender_data')
output_path = exporter.export_complete_session(
    tmr_data=sensors['tmr'],
    semg_data=sensors['semg'],
    imu_data=sensors['imu'],
    predictions=predictions,
    confidences=confidences,
    true_labels=labels,
    sampling_rate=50
)

print("\n" + "="*70)
print("✓ COMPLETE - ALL STEPS FINISHED!")
print("="*70)

print(f"\nResults:")
print(f"  Accuracy: {results['accuracy']:.2%}")
print(f"  Mean Latency: {results['mean_latency_ms']:.1f} ms")

print(f"\nGenerated Files:")
print(f"  • output/trained_model.pkl")
print(f"  • output/analysis/ (8 graphs + reports)")
print(f"  • output/blender_data/session_data.json")

print(f"\nNext Steps:")
print(f"  1. Check graphs: output/analysis/")
print(f"  2. Open Blender")
print(f"  3. Load script: Blender/advanced_animator.py")
print(f"  4. Press Spacebar to play animation")

print("\n" + "="*70)
print("NOTE: This used SYNTHETIC data (no download needed)")
print("For REAL data:")
print("  1. Go to: https://physionet.org/content/emgdb/1.0.0/")
print("  2. Download: emg_healthy_001.txt")
print("  3. Run: python 07_master_pipeline.py --data emg_healthy_001.txt")
print("="*70)

