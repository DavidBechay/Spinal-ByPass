import bpy
import json
import math
import os
from pathlib import Path
from mathutils import Euler, Vector, Quaternion

 
class AdvancedSpinalBypassAnimator:
    """
    Professional Blender animator for spinal bypass system
    """
    
    def __init__(self):
        self.armature = None
        self.bones = {}
        
        # Animation data
        self.session_data = None
        self.current_frame = 0
        self.total_frames = 0
        
        # Joint angles (smoothed)
        self.target_hip = 0.0
        self.target_knee = 0.0
        self.target_ankle = 0.0
        
        self.current_hip = 0.0
        self.current_knee = 0.0
        self.current_ankle = 0.0
        
        # Smoothing
        self.smoothing = 0.15  # Lower = smoother
        
        # UI Text objects
        self.ui_objects = {}
        
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        ADVANCED BLENDER SPINAL BYPASS ANIMATOR                       ║
║                                                                      ║
║            Production-Grade Human Visualization                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
    
    def setup(self):
        """Complete setup procedure"""
        print("\nSetup Procedure:")
        print("  1. Finding character armature...")
        if not self.find_armature():
            return False
        
        print("  2. Finding leg bones...")
        if not self.find_bones():
            return False
        
        print("  3. Loading session data...")
        if not self.load_session_data():
            return False
        
        print("  4. Setting up UI...")
        self.setup_ui()
        
        print("  5. Configuring scene...")
        self.configure_scene()
        
        print("\n✓ Setup complete!\n")
        return True
    
    def find_armature(self):
        """Find character armature"""
        armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
        
        if not armatures:
            print("\n❌ ERROR: No armature found!")
            print("\nTo fix:")
            print("  1. Go to https://www.mixamo.com")
            print("  2. Download a character (e.g., 'Josh', 'Amy')")
            print("  3. File → Import → FBX")
            print("  4. Select downloaded .fbx file")
            return False
        
        self.armature = armatures[0]
        print(f"    ✓ Found: {self.armature.name}")
        return True
    
    def find_bones(self):
        """Find leg bones in armature"""
        pose_bones = self.armature.pose.bones
        
        # Try multiple naming conventions
        bone_patterns = {
            'hip': ['LeftUpLeg', 'mixamorig:LeftUpLeg', 'UpperLeg.L', 'Thigh.L'],
            'knee': ['LeftLeg', 'mixamorig:LeftLeg', 'LowerLeg.L', 'Shin.L'],
            'ankle': ['LeftFoot', 'mixamorig:LeftFoot', 'Foot.L', 'Ankle.L'],
        }
        
        for joint, patterns in bone_patterns.items():
            for pattern in patterns:
                if pattern in pose_bones:
                    self.bones[joint] = pose_bones[pattern]
                    print(f"    ✓ {joint.capitalize()}: {pattern}")
                    break
            
            if joint not in self.bones:
                print(f"\n❌ ERROR: Could not find {joint} bone!")
                print(f"\nAvailable bones:")
                for bone in pose_bones:
                    print(f"  - {bone.name}")
                return False
        
        return True
    
    def load_session_data(self, filepath=None):
        """Load session data from JSON"""
        if filepath is None:
            # Try common locations
            blend_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
            
            possible_paths = [
                blend_dir / "output" / "blender_data" / "session_data.json",
                blend_dir / "blender_data" / "session_data.json",
                blend_dir / "session_data.json",
                Path.cwd() / "output" / "blender_data" / "session_data.json",
            ]
            
            for path in possible_paths:
                if path.exists():
                    filepath = path
                    break
        
        if filepath is None:
            print("\n❌ ERROR: session_data.json not found!")
            print("\nTo fix:")
            print("  1. Run Python pipeline:")
            print("     python 07_master_pipeline.py --data S1_E1_A1.mat")
            print("  2. This creates: output/blender_data/session_data.json")
            print("  3. Run this script again")
            return False
        
        with open(filepath, 'r') as f:
            self.session_data = json.load(f)
        
        self.total_frames = self.session_data['metadata']['num_frames']
        duration = self.session_data['metadata']['duration_seconds']
        
        print(f"    ✓ Loaded: {filepath}")
        print(f"    • Frames: {self.total_frames:,}")
        print(f"    • Duration: {duration:.1f} seconds")
        
        return True
    
    def setup_ui(self):
        """Create UI text displays"""
        # Add text objects for real-time info
        font_curve = bpy.data.curves.new(type="FONT", name="InfoText")
        font_curve.body = "Frame: 0"
        
        text_obj = bpy.data.objects.new(name="FrameCounter", object_data=font_curve)
        bpy.context.scene.collection.objects.link(text_obj)
        
        text_obj.location = (2, 0, 2)
        text_obj.scale = (0.3, 0.3, 0.3)
        
        self.ui_objects['frame_counter'] = text_obj
        
        print("    ✓ UI elements created")
    
    def configure_scene(self):
        """Configure Blender scene for animation"""
        # Set frame range
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = self.total_frames
        bpy.context.scene.render.fps = 25
        
        # Set current frame to start
        bpy.context.scene.frame_set(1)
        
        print(f"    ✓ Scene configured ({self.total_frames} frames @ 25 FPS)")
    
    def update_joints(self, frame_idx):
        """Update joint angles from data"""
        if frame_idx >= len(self.session_data['frames']):
            frame_idx = len(self.session_data['frames']) - 1
        
        frame_data = self.session_data['frames'][frame_idx]
        
        # Get target angles
        self.target_hip = frame_data['joints']['hip']
        self.target_knee = frame_data['joints']['knee']
        self.target_ankle = frame_data['joints']['ankle']
        
        # Smooth transition
        self.current_hip += (self.target_hip - self.current_hip) * self.smoothing
        self.current_knee += (self.target_knee - self.current_knee) * self.smoothing
        self.current_ankle += (self.target_ankle - self.current_ankle) * self.smoothing
        
        # Apply to bones
        if 'hip' in self.bones:
            self.bones['hip'].rotation_euler.x = math.radians(-self.current_hip)
        
        if 'knee' in self.bones:
            self.bones['knee'].rotation_euler.x = math.radians(self.current_knee)
        
        if 'ankle' in self.bones:
            self.bones['ankle'].rotation_euler.x = math.radians(self.current_ankle)
        
        # Update UI
        if 'frame_counter' in self.ui_objects:
            intent = frame_data['prediction']['intent']
            confidence = frame_data['prediction']['confidence']
            
            text = f"Frame: {frame_idx}\n"
            text += f"Intent: {intent}\n"
            text += f"Confidence: {confidence:.1%}\n"
            text += f"Hip: {self.current_hip:.1f}°\n"
            text += f"Knee: {self.current_knee:.1f}°\n"
            text += f"Ankle: {self.current_ankle:.1f}°"
            
            self.ui_objects['frame_counter'].data.body = text
    
    def bake_animation(self):
        """Bake animation to keyframes"""
        print("\nBaking animation to keyframes...")
        print("This may take a moment...")
        
        # Set rotation mode
        for bone_name, bone in self.bones.items():
            bone.rotation_mode = 'XYZ'
        
        # Keyframe every frame
        for frame_idx in range(self.total_frames):
            self.update_joints(frame_idx)
            
            # Set frame
            bpy.context.scene.frame_set(frame_idx + 1)
            
            # Keyframe bones
            for bone_name, bone in self.bones.items():
                bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx + 1)
            
            # Progress
            if (frame_idx + 1) % 100 == 0:
                print(f"  Baked {frame_idx + 1} / {self.total_frames} frames")
        
        print(f"✓ Animation baked ({self.total_frames} keyframes)")
    
    def play(self):
        """Start animation playback"""
        print("\nStarting animation...")
        print("Press SPACEBAR in Blender to play/pause")
        print("Press ESC to stop")
        
        bpy.ops.screen.animation_play()
    
    def render_animation(self, output_path="//animation.mp4"):
        """Render animation to video"""
        print(f"\nRendering animation to: {output_path}")
        print("This will take several minutes...")
        
        # Set render settings
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        bpy.context.scene.render.ffmpeg.codec = 'H264'
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        bpy.context.scene.render.filepath = output_path
        
        # Render
        bpy.ops.render.render(animation=True)
        
        print(f"✓ Rendering complete: {output_path}")


# ==============================================================================
# FRAME HANDLER (Auto-update during playback)
# ==============================================================================

animator_instance = None

def frame_change_handler(scene):
    """Called every frame change"""
    global animator_instance
    
    if animator_instance and animator_instance.session_data:
        frame_idx = scene.frame_current - 1  # Blender frames start at 1
        animator_instance.update_joints(frame_idx)


def register_handler(animator):
    """Register frame change handler"""
    global animator_instance
    animator_instance = animator
    
    # Remove existing handlers
    if frame_change_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(frame_change_handler)
    
    # Add handler
    bpy.app.handlers.frame_change_pre.append(frame_change_handler)
    
    print("✓ Frame handler registered")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    # Create animator
    animator = AdvancedSpinalBypassAnimator()
    
    # Setup
    if not animator.setup():
        print("\n❌ Setup failed. Please fix errors and try again.")
        return None
    
    # Register frame handler for live playback
    register_handler(animator)
    
    # Optional: Bake animation (uncomment if needed)
    # animator.bake_animation()
    
    print("\n" + "="*70)
    print("READY TO ANIMATE!")
    print("="*70)
    print("\nControls:")
    print("  SPACEBAR - Play/Pause animation")
    print("  ESC - Stop animation")
    print("  Arrow Keys - Step through frames")
    print("  Home - Jump to start")
    print("\nTo render video:")
    print("  1. Uncomment: animator.render_animation()")
    print("  2. Run script again")
    print("  3. Wait for rendering to complete")
    print("\n" + "="*70)
    
    return animator


# Run when script is executed
if __name__ == "__main__":
    animator = main()
