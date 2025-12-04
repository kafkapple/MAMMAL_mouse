#!/usr/bin/env python
"""
Blender Mesh Animation Script

Import mesh sequence into Blender for visualization and rendering.
Can be run from command line or inside Blender.

Usage:
    # Method 1: Run from command line
    blender --background --python scripts/blender_mesh_animation.py -- \
        --result_dir results/fitting/experiment_name \
        --output_video output.mp4

    # Method 2: Open Blender and run script in Scripting tab
    # Set RESULT_DIR variable below before running

    # Method 3: Import as Blender addon (copy to Blender scripts folder)

Requirements:
    - Blender 3.0+ (tested with 4.0)
    - OBJ mesh sequence from fitting results
"""

import os
import sys
import glob
import math
from pathlib import Path

# Configuration (modify these if running inside Blender directly)
RESULT_DIR = "/home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v01234_sparse3_20251203_235123"
OUTPUT_VIDEO = None  # Auto-generate if None
FPS = 30
RESOLUTION = (1920, 1080)
CAMERA_TYPE = "orbit"  # orbit, front, side, top


def get_args():
    """Parse command line arguments when running from CLI."""
    # Check if running from command line with arguments
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        # Use default configuration
        return {
            'result_dir': RESULT_DIR,
            'output_video': OUTPUT_VIDEO,
            'fps': FPS,
            'resolution': RESOLUTION,
            'camera_type': CAMERA_TYPE,
        }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_video", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--camera_type", type=str, default="orbit")
    parser.add_argument("--render", action="store_true", help="Render animation")

    args = parser.parse_args(argv)
    return vars(args)


def setup_blender_scene():
    """Initialize Blender scene with proper settings."""
    import bpy

    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear all collections except Scene Collection
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    # Create new collection for meshes
    mesh_collection = bpy.data.collections.new("Mesh_Sequence")
    bpy.context.scene.collection.children.link(mesh_collection)

    return mesh_collection


def import_mesh_sequence(obj_dir: str, collection) -> list:
    """Import OBJ mesh sequence into Blender."""
    import bpy

    # Find OBJ files
    obj_files = sorted(glob.glob(os.path.join(obj_dir, "step_2_frame_*.obj")))
    if not obj_files:
        obj_files = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))

    print(f"Found {len(obj_files)} OBJ files")

    mesh_objects = []
    for i, obj_file in enumerate(obj_files):
        # Import OBJ
        bpy.ops.wm.obj_import(filepath=obj_file)

        # Get imported object
        obj = bpy.context.selected_objects[0]
        obj.name = f"frame_{i:04d}"

        # Move to collection
        for col in obj.users_collection:
            col.objects.unlink(obj)
        collection.objects.link(obj)

        # Hide by default (will be animated)
        obj.hide_viewport = True
        obj.hide_render = True

        mesh_objects.append(obj)

        if (i + 1) % 10 == 0:
            print(f"Imported {i + 1}/{len(obj_files)} meshes")

    return mesh_objects


def setup_material(obj):
    """Apply material to mesh object."""
    import bpy

    mat = bpy.data.materials.new(name="MouseMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (0.8, 0.75, 0.7, 1.0)  # Light skin tone
    bsdf.inputs['Roughness'].default_value = 0.4
    bsdf.inputs['Specular IOR Level'].default_value = 0.5

    # Create output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def setup_lighting():
    """Set up three-point lighting."""
    import bpy

    # Key light
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 5))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    key_light.data.size = 2

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 3))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 200
    fill_light.data.size = 3

    # Rim light
    bpy.ops.object.light_add(type='AREA', location=(0, 4, 4))
    rim_light = bpy.context.active_object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = 300
    rim_light.data.size = 2

    return [key_light, fill_light, rim_light]


def setup_camera(mesh_objects: list, camera_type: str = "orbit"):
    """Set up camera with optional orbit animation."""
    import bpy

    # Get bounding box of all meshes
    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3

    for obj in mesh_objects[:1]:  # Use first frame for reference
        obj.hide_viewport = False
        bbox = [obj.matrix_world @ bpy.mathutils.Vector(corner) for corner in obj.bound_box]
        obj.hide_viewport = True

        for corner in bbox:
            for i in range(3):
                min_coords[i] = min(min_coords[i], corner[i])
                max_coords[i] = max(max_coords[i], corner[i])

    # Calculate center and distance
    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
    size = max(max_coords[i] - min_coords[i] for i in range(3))
    distance = size * 2.5

    # Create camera
    bpy.ops.object.camera_add(location=(center[0], center[1] - distance, center[2] + distance * 0.3))
    camera = bpy.context.active_object
    camera.name = "Animation_Camera"

    # Point camera at center
    direction = bpy.mathutils.Vector(center) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = camera

    # Setup orbit animation if requested
    if camera_type == "orbit":
        # Create empty at center for orbit target
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
        orbit_target = bpy.context.active_object
        orbit_target.name = "Camera_Target"

        # Parent camera to empty
        camera.parent = orbit_target
        camera.location = (0, -distance, distance * 0.3)

        # Point camera at target
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = orbit_target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        # Animate rotation
        orbit_target.rotation_euler = (0, 0, 0)
        orbit_target.keyframe_insert(data_path="rotation_euler", frame=1)

        orbit_target.rotation_euler = (0, 0, math.pi * 2)
        orbit_target.keyframe_insert(data_path="rotation_euler", frame=len(mesh_objects))

        # Set interpolation to linear
        for fcurve in orbit_target.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'

    return camera


def setup_animation(mesh_objects: list, fps: int = 30):
    """Set up frame-by-frame mesh visibility animation."""
    import bpy

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(mesh_objects)
    scene.render.fps = fps

    for i, obj in enumerate(mesh_objects):
        frame = i + 1

        # Apply material
        setup_material(obj)

        # Hide before this frame
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame - 1)
        obj.keyframe_insert(data_path="hide_render", frame=frame - 1)

        # Show on this frame
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)

        # Hide after this frame
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame + 1)
        obj.keyframe_insert(data_path="hide_render", frame=frame + 1)

    print(f"Animation setup complete: {len(mesh_objects)} frames at {fps} FPS")


def setup_render_settings(resolution: tuple, output_path: str):
    """Configure render settings."""
    import bpy

    scene = bpy.context.scene

    # Resolution
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100

    # Output settings
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.filepath = output_path

    # Render engine
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 64  # Adjust for quality/speed tradeoff

    # If no GPU, use EEVEE
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type == 'CUDA':
                device.use = True
    except Exception:
        print("GPU not available, using EEVEE renderer")
        scene.render.engine = 'BLENDER_EEVEE'

    print(f"Render settings: {resolution[0]}x{resolution[1]}, output: {output_path}")


def render_animation():
    """Render the animation to video."""
    import bpy
    print("Starting render...")
    bpy.ops.render.render(animation=True)
    print("Render complete!")


def save_blend_file(output_path: str):
    """Save the Blender file for later editing."""
    import bpy
    blend_path = output_path.replace('.mp4', '.blend')
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Blender file saved: {blend_path}")


def main():
    import bpy

    args = get_args()

    result_dir = Path(args['result_dir'])
    obj_dir = result_dir / "obj"

    if not obj_dir.exists():
        print(f"Error: OBJ directory not found: {obj_dir}")
        return

    # Output path
    if args.get('output_video'):
        output_path = args['output_video']
    else:
        output_path = str(result_dir / "animation_blender.mp4")

    resolution = tuple(args.get('resolution', RESOLUTION))
    fps = args.get('fps', FPS)
    camera_type = args.get('camera_type', CAMERA_TYPE)

    print(f"\n{'='*50}")
    print("Blender Mesh Animation Setup")
    print(f"{'='*50}")
    print(f"Result directory: {result_dir}")
    print(f"Output: {output_path}")
    print(f"Resolution: {resolution}")
    print(f"FPS: {fps}")
    print(f"Camera: {camera_type}")
    print(f"{'='*50}\n")

    # Setup scene
    print("Setting up scene...")
    collection = setup_blender_scene()

    # Import meshes
    print("Importing mesh sequence...")
    mesh_objects = import_mesh_sequence(str(obj_dir), collection)

    if not mesh_objects:
        print("Error: No meshes imported")
        return

    # Setup lighting
    print("Setting up lighting...")
    setup_lighting()

    # Setup camera
    print("Setting up camera...")
    setup_camera(mesh_objects, camera_type)

    # Setup animation
    print("Setting up animation...")
    setup_animation(mesh_objects, fps)

    # Setup render settings
    print("Configuring render settings...")
    setup_render_settings(resolution, output_path)

    # Save blend file
    save_blend_file(output_path)

    # Render if requested from command line
    if args.get('render'):
        render_animation()

    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print("\nTo render animation:")
    print("  1. Open the saved .blend file in Blender")
    print("  2. Press Ctrl+F12 to render animation")
    print("  OR run with --render flag from command line")
    print("\nTo preview:")
    print("  1. Press Space to play animation in viewport")
    print("  2. Use Timeline to scrub through frames")


if __name__ == "__main__":
    main()
