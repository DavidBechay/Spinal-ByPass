"""
Outputs:
1. Performance over time (accuracy, confidence, latency)
2. Confusion matrix with per-class metrics
3. ROC curves (multi-class)
4. Feature importance visualization
5. TMR sensor activity heatmap
6. Joint angle trajectories
7. Movement distribution
8. Error analysis
9. Comprehensive PDF report
""" 

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and visualization
    """
    
    def __init__(self, output_dir: str = "analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Analysis results
        self.metrics = {}
        self.figures_generated = []

    def _accuracy_target_line(self) -> str:
        """Prefer held-out test accuracy for ≥85%% compliance when provided."""
        if self.metrics.get("held_out_test_accuracy") is not None:
            v = self.metrics["held_out_test_accuracy"]
            ok = v >= 0.85
            return f"{'✓ PASS' if ok else '✗ FAIL'}  (held-out test: {v:.2%})"
        v = self.metrics["accuracy"]
        ok = v >= 0.85
        return f"{'✓ PASS' if ok else '✗ FAIL'}  (full-run: {v:.2%})"
    
    def analyze_complete_session(self,
                                 true_labels: np.ndarray,
                                 predictions: np.ndarray,
                                 confidences: np.ndarray,
                                 latencies: np.ndarray,
                                 tmr_data: np.ndarray,
                                 imu_data: np.ndarray,
                                 feature_importance: Optional[np.ndarray] = None,
                                 feature_names: Optional[List[str]] = None,
                                 policy_metrics: Optional[Dict] = None,
                                 held_out_accuracy: Optional[float] = None) -> Dict:
        """
        Complete analysis pipeline
        
        Args:
            true_labels: (N,) ground truth
            predictions: (N,) predicted labels
            confidences: (N,) confidence scores
            latencies: (N,) latency measurements (ms)
            tmr_data: (N, 8) TMR readings
            imu_data: (N, 3) joint angles
            feature_importance: (features,) importance scores
            feature_names: Feature names
        
        Returns:
            Dictionary with all metrics
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*70)
        
        N = len(true_labels)
        
        # Compute metrics
        print("\n1. Computing performance metrics...")
        self.metrics = self._compute_all_metrics(
            true_labels, predictions, confidences, latencies
        )

        # Add TMR sensor quality diagnostics
        tmr_quality = self._analyze_tmr_quality(tmr_data)
        self.metrics.update(tmr_quality)

        if policy_metrics:
            self.metrics["policy"] = policy_metrics
            if policy_metrics.get("mean_tmr_snr_linear") is not None:
                self.metrics["tmr_snr_linear_effective"] = policy_metrics["mean_tmr_snr_linear"]

        if held_out_accuracy is not None:
            self.metrics["held_out_test_accuracy"] = float(held_out_accuracy)

        # Generate all visualizations
        print("\n2. Generating visualizations...")
        
        self.plot_performance_over_time(
            true_labels, predictions, confidences, latencies
        )
        
        self.plot_confusion_matrix(true_labels, predictions)
        
        self.plot_latency_analysis(latencies)
        
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance, feature_names)
        
        self.plot_tmr_heatmap(tmr_data)
        
        self.plot_joint_trajectories(imu_data)
        
        self.plot_movement_distribution(true_labels, predictions)
        
        self.plot_error_analysis(true_labels, predictions, confidences)
        
        # Generate reports
        print("\n3. Generating reports...")
        self.generate_text_report()
        self.generate_json_metrics()
        
        print(f"\n{'='*70}")
        print(f"✓ ANALYSIS COMPLETE")
        print(f"  {len(self.figures_generated)} figures generated")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*70}")
        
        return self.metrics
    
    def _compute_all_metrics(self, true_labels, predictions, confidences, latencies):
        """Compute all performance metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix
        )
        
        metrics = {}
        
        # Overall metrics
        metrics['num_samples'] = len(true_labels)
        metrics['accuracy'] = accuracy_score(true_labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['support_per_class'] = support.tolist()
        
        # Averaged metrics
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        # Confidence metrics
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['std_confidence'] = float(np.std(confidences))
        metrics['min_confidence'] = float(np.min(confidences))
        metrics['max_confidence'] = float(np.max(confidences))
        
        # Latency metrics
        metrics['mean_latency_ms'] = float(np.mean(latencies))
        metrics['std_latency_ms'] = float(np.std(latencies))
        metrics['min_latency_ms'] = float(np.min(latencies))
        metrics['max_latency_ms'] = float(np.max(latencies))
        metrics['p50_latency_ms'] = float(np.percentile(latencies, 50))
        metrics['p95_latency_ms'] = float(np.percentile(latencies, 95))
        metrics['p99_latency_ms'] = float(np.percentile(latencies, 99))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Error rate
        metrics['error_rate'] = 1 - metrics['accuracy']
        
        return metrics
    
    def _analyze_tmr_quality(self, tmr_data):
        """Computes TMR SNR and crosstalk diagnostics"""
        tmr_data = np.asarray(tmr_data)
        if tmr_data.ndim != 2 or tmr_data.shape[1] != 8:
            return {
                'tmr_snr': None,
                'tmr_crosstalk_rate': None,
                'tmr_left_right_balance': None,
            }

        # Estimated SNR in simple terms: signal RMS vs noise RMS
        signal_rms = np.sqrt(np.mean(tmr_data ** 2))
        noise_rms = np.std(tmr_data - np.mean(tmr_data, axis=0))
        snr = float(signal_rms / (noise_rms + 1e-12))

        # Common-mode event rate: simultaneous high derivative across many sensors
        derivative = np.abs(np.diff(tmr_data, axis=0))
        threshold = np.percentile(derivative, 95)
        simultaneous = np.sum(np.sum(derivative >= threshold, axis=1) >= 6)
        crosstalk_rate = float(simultaneous / max(1, derivative.shape[0]))

        # Simple symmetry check (L/R balance)
        left = tmr_data[:, [0,2,4,6]]
        right = tmr_data[:, [1,3,5,7]]
        balance = float(np.mean(np.abs(np.mean(left, axis=1) - np.mean(right, axis=1))))

        return {
            'tmr_snr': snr,
            'tmr_crosstalk_rate': crosstalk_rate,
            'tmr_left_right_balance': balance,
        }
    
    def plot_performance_over_time(self, true_labels, predictions, 
                                   confidences, latencies, window=50):
        """Plot accuracy, confidence, and latency over time"""
        N = len(true_labels)
        time = np.arange(N) * 0.02  # 50 Hz
        
        # Rolling metrics
        correct = (predictions == true_labels).astype(float)
        rolling_acc = np.convolve(correct, np.ones(window)/window, mode='valid')
        rolling_conf = np.convolve(confidences, np.ones(window)/window, mode='valid')
        rolling_time = time[:len(rolling_acc)]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Accuracy
        axes[0].plot(rolling_time, rolling_acc * 100, linewidth=2, 
                    color='#2ecc71', label='Rolling Accuracy')
        axes[0].axhline(y=85, color='r', linestyle='--', alpha=0.5, 
                       label='Target (85%)')
        axes[0].fill_between(rolling_time, 0, rolling_acc * 100, 
                            alpha=0.3, color='#2ecc71')
        axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[0].set_ylim([0, 105])
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('System Performance Over Time', 
                         fontsize=14, fontweight='bold', pad=15)
        
        # Confidence
        axes[1].plot(rolling_time, rolling_conf * 100, linewidth=2, 
                    color='#3498db', label='Rolling Confidence')
        axes[1].fill_between(rolling_time, 0, rolling_conf * 100, 
                            alpha=0.3, color='#3498db')
        axes[1].set_ylabel('Confidence (%)', fontweight='bold')
        axes[1].set_ylim([0, 105])
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        # Latency
        rolling_lat = np.convolve(latencies, np.ones(window)/window, mode='valid')
        axes[2].plot(rolling_time, rolling_lat, linewidth=2, 
                    color='#9b59b6', label='Rolling Latency')
        axes[2].axhline(y=80, color='g', linestyle='--', alpha=0.5, 
                       label='Target (80 ms)')
        axes[2].fill_between(rolling_time, 0, rolling_lat, 
                            alpha=0.3, color='#9b59b6')
        axes[2].set_xlabel('Time (seconds)', fontweight='bold')
        axes[2].set_ylabel('Latency (ms)', fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'performance_over_time.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix with normalization"""
        from sklearn.metrics import confusion_matrix
        
        labels = sorted(np.unique(np.concatenate([true_labels, predictions])))
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax1)
        ax1.set_xlabel('Predicted', fontweight='bold')
        ax1.set_ylabel('True', fontweight='bold')
        ax1.set_title('Confusion Matrix (Counts)', fontweight='bold')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Proportion'}, ax=ax2)
        ax2.set_xlabel('Predicted', fontweight='bold')
        ax2.set_ylabel('True', fontweight='bold')
        ax2.set_title('Confusion Matrix (Normalized)', fontweight='bold')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'confusion_matrix.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def plot_latency_analysis(self, latencies):
        """Detailed latency analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        axes[0, 0].hist(latencies, bins=50, color='#9b59b6', 
                       alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=np.mean(latencies), color='r', 
                          linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(latencies):.1f} ms')
        axes[0, 0].axvline(x=np.percentile(latencies, 95), color='orange', 
                          linestyle='--', linewidth=2,
                          label=f'95th: {np.percentile(latencies, 95):.1f} ms')
        axes[0, 0].axvline(x=80, color='g', linestyle='--', 
                          linewidth=2, alpha=0.5, label='Target: 80 ms')
        axes[0, 0].set_xlabel('Latency (ms)', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Latency Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        bp = axes[0, 1].boxplot(latencies, vert=True, patch_artist=True,
                                boxprops=dict(facecolor='#9b59b6', alpha=0.7),
                                medianprops=dict(color='red', linewidth=2))
        axes[0, 1].axhline(y=80, color='g', linestyle='--', 
                          linewidth=2, alpha=0.5, label='Target')
        axes[0, 1].set_ylabel('Latency (ms)', fontweight='bold')
        axes[0, 1].set_title('Latency Box Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].legend()
        
        # CDF
        sorted_lat = np.sort(latencies)
        cdf = np.arange(1, len(sorted_lat)+1) / len(sorted_lat)
        axes[1, 0].plot(sorted_lat, cdf * 100, linewidth=2, color='#9b59b6')
        axes[1, 0].axvline(x=80, color='g', linestyle='--', 
                          linewidth=2, alpha=0.5, label='Target: 80 ms')
        axes[1, 0].set_xlabel('Latency (ms)', fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Probability (%)', fontweight='bold')
        axes[1, 0].set_title('Cumulative Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        values = [np.percentile(latencies, p) for p in percentiles]
        axes[1, 1].bar(range(len(percentiles)), values, 
                      color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[1, 1].axhline(y=80, color='g', linestyle='--', 
                          linewidth=2, alpha=0.5, label='Target')
        axes[1, 1].set_xticks(range(len(percentiles)))
        axes[1, 1].set_xticklabels([f'P{p}' for p in percentiles])
        axes[1, 1].set_ylabel('Latency (ms)', fontweight='bold')
        axes[1, 1].set_title('Latency Percentiles', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'latency_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def plot_feature_importance(self, importance, feature_names=None):
        """Plot top feature importances"""
        top_n = 20
        
        # Get top features
        sorted_idx = np.argsort(importance)[-top_n:]
        
        if feature_names:
            labels = [feature_names[i] for i in sorted_idx]
        else:
            labels = [f'Feature {i}' for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(top_n), importance[sorted_idx], 
               color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important Features', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'feature_importance.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def plot_tmr_heatmap(self, tmr_data):
        """TMR sensor activity heatmap"""
        # Downsample for visualization
        tmr_viz = tmr_data[::10].T
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.imshow(tmr_viz, aspect='auto', cmap='hot', interpolation='bilinear', vmin=-4, vmax=4)
        
        sensor_labels = ['L1 L', 'L1 R', 'L2 L', 'L2 R', 
                        'L3 L', 'L3 R', 'L4 L', 'L4 R']
        ax.set_yticks(range(8))
        ax.set_yticklabels(sensor_labels)
        ax.set_xlabel('Time (samples × 10)', fontweight='bold')
        ax.set_ylabel('TMR Sensor', fontweight='bold')
        ax.set_title('TMR Sensor Activity Heatmap', 
                    fontweight='bold', fontsize=14, pad=15)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Strength (nT)', fontweight='bold')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'tmr_heatmap.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def plot_joint_trajectories(self, imu_data):
        """Joint angle trajectories"""
        N = imu_data.shape[0]
        time = np.arange(N) * 0.02
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        joints = ['Hip', 'Knee', 'Ankle']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for i, (ax, joint, color) in enumerate(zip(axes, joints, colors)):
            ax.plot(time, imu_data[:, i], linewidth=2, color=color, label=joint)
            ax.fill_between(time, imu_data[:, i], alpha=0.3, color=color)
            ax.set_ylabel(f'{joint} Angle (°)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.legend(loc='upper right')
        
        axes[2].set_xlabel('Time (seconds)', fontweight='bold')
        axes[0].set_title('Joint Angle Trajectories', 
                         fontweight='bold', fontsize=14, pad=15)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'joint_trajectories.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def plot_movement_distribution(self, true_labels, predictions):
        """Movement class distribution"""
        labels = sorted(np.unique(np.concatenate([true_labels, predictions])))
        
        true_counts = [np.sum(true_labels == label) for label in labels]
        pred_counts = [np.sum(predictions == label) for label in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width/2, true_counts, width, label='True', 
              color='#3498db', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, pred_counts, width, label='Predicted', 
              color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Movement Class', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Movement Distribution', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'movement_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def plot_error_analysis(self, true_labels, predictions, confidences):
        """Error analysis - where does model fail?"""
        errors = true_labels != predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Error rate over time
        window = 50
        error_rate = np.convolve(errors.astype(float), 
                                np.ones(window)/window, mode='valid')
        time = np.arange(len(error_rate)) * 0.02
        
        axes[0].plot(time, error_rate * 100, linewidth=2, color='#e74c3c')
        axes[0].fill_between(time, 0, error_rate * 100, alpha=0.3, color='#e74c3c')
        axes[0].set_xlabel('Time (seconds)', fontweight='bold')
        axes[0].set_ylabel('Error Rate (%)', fontweight='bold')
        axes[0].set_title('Error Rate Over Time', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence for correct vs incorrect
        correct_conf = confidences[~errors]
        incorrect_conf = confidences[errors]
        
        axes[1].hist([correct_conf, incorrect_conf], bins=30, 
                    label=['Correct', 'Incorrect'],
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, 
                    edgecolor='black')
        axes[1].set_xlabel('Confidence', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Confidence Distribution (Correct vs Incorrect)', 
                         fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'error_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures_generated.append(str(filepath))
        print(f"  ✓ Saved: {filepath.name}")
    
    def generate_text_report(self):
        """Generate comprehensive text report"""
        report = f"""
SPINAL BYPASS SYSTEM - COMPREHENSIVE PERFORMANCE REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SESSION INFORMATION
{'-'*70}
Total Samples:          {self.metrics['num_samples']:,}
Duration:               {self.metrics['num_samples'] * 0.02:.1f} seconds
Sample Rate:            50 Hz

ACCURACY METRICS
{'-'*70}
Overall Accuracy:       {self.metrics['accuracy']:.2%}  (predictions on full run — can include train overlap)
"""
        if self.metrics.get("held_out_test_accuracy") is not None:
            report += f"""Held-out Test Acc.:   {self.metrics['held_out_test_accuracy']:.2%}  (unseen stratified test set from training)
"""
        report += f"""\
Precision (macro):      {self.metrics['precision_macro']:.2%}
Recall (macro):         {self.metrics['recall_macro']:.2%}
F1 Score (macro):       {self.metrics['f1_macro']:.2%}

CONFIDENCE METRICS
{'-'*70}
Mean Confidence:        {self.metrics['mean_confidence']:.2%}
Std Confidence:         {self.metrics['std_confidence']:.2%}
Min Confidence:         {self.metrics['min_confidence']:.2%}
Max Confidence:         {self.metrics['max_confidence']:.2%}

LATENCY METRICS
{'-'*70}
Mean Latency:           {self.metrics['mean_latency_ms']:.1f} ms
Std Latency:            {self.metrics['std_latency_ms']:.1f} ms
Min Latency:            {self.metrics['min_latency_ms']:.1f} ms
Max Latency:            {self.metrics['max_latency_ms']:.1f} ms
50th Percentile:        {self.metrics['p50_latency_ms']:.1f} ms
95th Percentile:        {self.metrics['p95_latency_ms']:.1f} ms
99th Percentile:        {self.metrics['p99_latency_ms']:.1f} ms

TARGET COMPLIANCE
{'-'*70}
Accuracy ≥85%:          {self._accuracy_target_line()}
Latency <80ms (mean):   {'✓ PASS' if self.metrics['mean_latency_ms'] < 80 else '✗ FAIL'}
Latency <120ms (P99):   {'✓ PASS' if self.metrics['p99_latency_ms'] < 120 else '✗ FAIL'}

TMR SENSOR QUALITY
{'-'*70}
TMR SNR (est):         {self.metrics.get('tmr_snr', 0):.2f} {'(PASS)' if self.metrics.get('tmr_snr', 0) >= 10 else '(FAIL)'}
TMR SNR linear (eff.): {self.metrics.get('tmr_snr_linear_effective', self.metrics.get('tmr_snr', 0)) or 0:.2f}  (policy window)
TMR Crosstalk rate:    {self.metrics.get('tmr_crosstalk_rate', 0):.3f}
L/R imbalance:         {self.metrics.get('tmr_left_right_balance', 0):.4f}
"""
        pol = self.metrics.get("policy")
        if pol:
            acc_c = pol.get("accuracy_on_committed")
            acc_c_str = f"{acc_c:.2%}" if acc_c is not None else "n/a"
            report += f"""
POLICY (confidence + TMR SNR gating)
{'-'*70}
Abstention rate:       {pol.get('abstention_rate', 0):.2%}
Accuracy (committed):  {acc_c_str}
Mean TMR SNR (lin.):   {pol.get('mean_tmr_snr_linear', 0):.2f}
Pass mean latency:     {pol.get('pass_mean_latency')}
Pass P99 latency:      {pol.get('pass_p99_latency')}
Pass TMR SNR:          {pol.get('pass_tmr_snr')}
Pass acc. (committed): {pol.get('pass_accuracy_committed')}
"""

        report += f"""
GENERATED FIGURES
{'-'*70}
"""
        for fig in self.figures_generated:
            report += f"  • {Path(fig).name}\n"
        
        report += f"\n{'='*70}\n"
        
        # Save
        filepath = self.output_dir / 'performance_report.txt'
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(report)
        
        print(f"  ✓ Saved: {filepath.name}")
        
        # Also print to console
        print(report)
    
    def generate_json_metrics(self):
        """Save metrics as JSON"""
        filepath = self.output_dir / 'performance_metrics.json'

        def _json_safe(obj):
            if isinstance(obj, dict):
                return {str(k): _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_safe(x) for x in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (bool, str, int, float)) or obj is None:
                return obj
            return str(obj)

        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(_json_safe(self.metrics), f, indent=2)
        
        print(f"  ✓ Saved: {filepath.name}")


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            ADVANCED DATA ANALYSIS & VISUALIZATION                    ║
║                                                                      ║
║          Publication-Quality Performance Analysis                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nGenerated Outputs:")
    print("  1. performance_over_time.png - Accuracy/confidence/latency trends")
    print("  2. confusion_matrix.png - Classification performance")
    print("  3. latency_analysis.png - Detailed latency statistics")
    print("  4. feature_importance.png - Top contributing features")
    print("  5. tmr_heatmap.png - Sensor activity patterns")
    print("  6. joint_trajectories.png - Movement kinematics")
    print("  7. movement_distribution.png - Class balance")
    print("  8. error_analysis.png - Failure mode analysis")
    print("  9. performance_report.txt - Comprehensive text report")
    print(" 10. performance_metrics.json - Machine-readable metrics")
