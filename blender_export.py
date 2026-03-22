"""
BLENDER DATA EXPORT
Export simulation data in Blender-compatible format

Outputs:
- session_data.json - Complete session for playback
- frame_by_frame/ - Individual frame files
- metadata.json - Dataset information
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class BlenderDataExporter:
    """
    Export data for Blender visualization
    """
    
    def __init__(self, output_dir: str = "blender_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_complete_session(self,
                               tmr_data: np.ndarray,
                               semg_data: np.ndarray,
                               imu_data: np.ndarray,
                               predictions: np.ndarray,
                               confidences: np.ndarray,
                               true_labels: Optional[np.ndarray] = None,
                               sampling_rate: int = 50) -> Path:
        """
        Export complete session to single JSON file
        
        Args:
            tmr_data: (N, 8) TMR readings
            semg_data: (N, 64) sEMG channels
            imu_data: (N, 3) joint angles
            predictions: (N,) predicted movements
            confidences: (N,) confidence scores
            true_labels: (N,) ground truth (optional)
            sampling_rate: Hz (default 50 for smooth animation)
        
        Returns:
            Path to saved JSON file
        """
        print(f"\n{'='*70}")
        print("EXPORTING DATA FOR BLENDER")
        print(f"{'='*70}")
        
        N = len(predictions)
        
        # Create session data structure
        session_data = {
            "metadata": {
                "num_frames": N,
                "duration_seconds": N / sampling_rate,
                "sampling_rate_hz": sampling_rate,
                "exported_at": datetime.now().isoformat(),
                "channels": {
                    "tmr": tmr_data.shape[1],
                    "semg": semg_data.shape[1],
                    "imu": imu_data.shape[1]
                }
            },
            "frames": []
        }
        
        print(f"\nProcessing {N:,} frames...")
        
        # Create label encoding for string predictions (MEILoD support)
        label_to_id = {}
        if predictions.dtype == 'object':  # String predictions
            unique_labels = list(set(predictions))
            label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            print(f"  Label mapping: {label_to_id}")
        
        # Process each frame
        for i in range(N):
            # Handle both numeric and string predictions
            pred_value = predictions[i]
            if isinstance(pred_value, (str, np.str_)):
                pred_id = label_to_id.get(str(pred_value), 0)
                pred_label = str(pred_value)
            else:
                pred_id = int(pred_value)
                pred_label = f"class_{pred_id}"
            
            # Handle both numeric and string true labels
            true_id = None
            if true_labels is not None:
                true_value = true_labels[i]
                if isinstance(true_value, (str, np.str_)):
                    true_id = label_to_id.get(str(true_value), 0)
                else:
                    true_id = int(true_value)
            
            frame = {
                "frame": i,
                "time": i / sampling_rate,
                
                "sensors": {
                    "tmr": {
                        "L1_left": float(tmr_data[i, 0]),
                        "L1_right": float(tmr_data[i, 1]),
                        "L2_left": float(tmr_data[i, 2]),
                        "L2_right": float(tmr_data[i, 3]),
                        "L3_left": float(tmr_data[i, 4]),
                        "L3_right": float(tmr_data[i, 5]),
                        "L4_left": float(tmr_data[i, 6]),
                        "L4_right": float(tmr_data[i, 7]),
                    },
                    "semg": {
                        "iliopsoas": float(np.sqrt(np.mean(semg_data[i, 0:16]**2))),
                        "quadriceps": float(np.sqrt(np.mean(semg_data[i, 16:32]**2))),
                        "hamstrings": float(np.sqrt(np.mean(semg_data[i, 32:48]**2))),
                        "tibialis": float(np.sqrt(np.mean(semg_data[i, 48:64]**2))),
                    },
                    "imu": {
                        "hip_angle": float(imu_data[i, 0]),
                        "knee_angle": float(imu_data[i, 1]),
                        "ankle_angle": float(imu_data[i, 2]),
                    }
                },
                
                "prediction": {
                    "intent": pred_id,
                    "intent_label": pred_label,
                    "confidence": float(confidences[i]),
                },
                
                "joints": {
                    "hip": float(imu_data[i, 0]),
                    "knee": float(imu_data[i, 1]),
                    "ankle": float(imu_data[i, 2]),
                }
            }
            
            # Add ground truth if available
            if true_labels is not None:
                frame["ground_truth"] = true_id
                frame["ground_truth_label"] = str(true_labels[i]) if isinstance(true_labels[i], (str, np.str_)) else f"class_{true_id}"
            
            session_data["frames"].append(frame)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1:,} / {N:,} frames ({(i+1)/N*100:.1f}%)")
        
        # Save
        output_path = self.output_dir / "session_data.json"
        
        print(f"\nSaving to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n✓ Export complete:")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Frames: {N:,}")
        print(f"  Duration: {N/sampling_rate:.1f} seconds")
        
        return output_path
    
    def export_frame_by_frame(self,
                              tmr_data: np.ndarray,
                              semg_data: np.ndarray,
                              imu_data: np.ndarray,
                              predictions: np.ndarray,
                              confidences: np.ndarray):
        """
        Export individual JSON file per frame
        
        Useful for:
        - Large datasets (avoid huge single file)
        - Frame-by-frame processing in Blender
        - Debugging specific frames
        """
        print(f"\n{'='*70}")
        print("EXPORTING FRAME-BY-FRAME DATA")
        print(f"{'='*70}")
        
        N = len(predictions)
        
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"\nExporting {N:,} individual frame files...")
        
        for i in range(N):
            frame_data = {
                "frame": i,
                "time": i * 0.02,  # 50 Hz
                "joints": {
                    "hip": float(imu_data[i, 0]),
                    "knee": float(imu_data[i, 1]),
                    "ankle": float(imu_data[i, 2]),
                },
                "prediction": predictions[i],
                "confidence": float(confidences[i]),
            }
            
            filepath = frames_dir / f"frame_{i:06d}.json"
            with open(filepath, 'w') as f:
                json.dump(frame_data, f)
            
            if (i + 1) % 100 == 0:
                print(f"  Exported {i+1:,} / {N:,} frames")
        
        print(f"\n✓ Frame-by-frame export complete:")
        print(f"  Directory: {frames_dir}")
        print(f"  Files: {N:,}")
    
    def export_metadata(self, dataset_info: Dict):
        """
        Export metadata about the dataset
        
        Args:
            dataset_info: Dictionary with dataset information
        """
        metadata = {
            "exported_at": datetime.now().isoformat(),
            "dataset": dataset_info,
            "format_version": "1.0",
            "blender_compatible": True,
        }
        
        filepath = self.output_dir / "metadata.json"
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {filepath}")


class WebSocketDataStreamer:
    """
    Stream data to Blender via WebSocket (real-time mode)
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
    
    async def stream_session(self,
                            tmr_data: np.ndarray,
                            semg_data: np.ndarray,
                            imu_data: np.ndarray,
                            predictions: np.ndarray,
                            confidences: np.ndarray,
                            fps: int = 50):
        """
        Stream data to Blender in real-time
        
        Args:
            tmr_data: (N, 8)
            semg_data: (N, 64)
            imu_data: (N, 3)
            predictions: (N,)
            confidences: (N,)
            fps: Streaming frame rate
        """
        import asyncio
        import websockets
        
        N = len(predictions)
        frame_period = 1.0 / fps
        
        print(f"\n{'='*70}")
        print("WEBSOCKET STREAMING TO BLENDER")
        print(f"{'='*70}")
        print(f"Server: ws://{self.host}:{self.port}")
        print(f"Frames: {N:,}")
        print(f"FPS: {fps}")
        
        clients = set()
        
        async def handler(websocket, path):
            clients.add(websocket)
            print(f"✓ Blender connected from {websocket.remote_address}")
            
            try:
                await websocket.wait_closed()
            finally:
                clients.remove(websocket)
                print(f"✗ Blender disconnected")
        
        async def broadcast():
            # Wait for connection
            print("\nWaiting for Blender to connect...")
            while not clients:
                await asyncio.sleep(0.1)
            
            print("✓ Blender connected! Starting stream...\n")
            
            for i in range(N):
                message = {
                    "frame": i,
                    "time": i / fps,
                    "joints": {
                        "hip": float(imu_data[i, 0]),
                        "knee": float(imu_data[i, 1]),
                        "ankle": float(imu_data[i, 2]),
                    },
                    "prediction": predictions[i],
                    "confidence": float(confidences[i]),
                }
                
                message_json = json.dumps(message)
                
                # Send to all connected clients
                if clients:
                    await asyncio.gather(
                        *[client.send(message_json) for client in clients],
                        return_exceptions=True
                    )
                
                if i % 50 == 0:
                    print(f"Streamed {i} frames...")
                
                await asyncio.sleep(frame_period)
            
            print(f"\n✓ Streaming complete ({N} frames)")
        
        # Start server
        async with websockets.serve(handler, self.host, self.port):
            await broadcast()


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                  BLENDER DATA EXPORT                                 ║
║                                                                      ║
║         Export Simulation Data for Blender Visualization             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nExport Modes:")
    print("  1. Complete Session (single JSON file)")
    print("     - Best for most use cases")
    print("     - Blender loads entire session at once")
    print("     - File size: ~1-5 MB for 500 frames")
    
    print("\n  2. Frame-by-Frame (individual JSON files)")
    print("     - For very large datasets")
    print("     - Blender loads one frame at a time")
    print("     - More files, but smaller each")
    
    print("\n  3. WebSocket Streaming (real-time)")
    print("     - Live streaming to Blender")
    print("     - Requires websockets library")
    print("     - Best for live demos")
    
    print("\nExample usage:")
    print("""
from blender_export import BlenderDataExporter

# Export complete session
exporter = BlenderDataExporter(output_dir='blender_data')

output_path = exporter.export_complete_session(
    tmr_data=tmr,
    semg_data=semg,
    imu_data=imu,
    predictions=predictions,
    confidences=confidences
)

# This creates: blender_data/session_data.json
# Blender script will load this file
    """)
