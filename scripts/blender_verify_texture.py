"""
Blender headless texture verification.

Loads 3 sample textured OBJs (first/mid/last), renders each from a fixed camera,
and saves PNG images to verify texture mapping is correct.

Run with:
    /home/joon/blender-4.0.2-linux-x64/blender --background --python \
        /home/joon/dev/MAMMAL_mouse/scripts/blender_verify_texture.py

All paths are ABSOLUTE to avoid CWD dependency in headless mode.
"""

import bpy
import os
import sys

# ---- Absolute paths (headless Blender ignores CWD) ----
PROJECT_ROOT = "/home/joon/dev/MAMMAL_mouse"
OBJ_DIR = os.path.join(PROJECT_ROOT, "results/fitting/production_3600_slerp/obj_textured")
TEXTURE_PNG = os.path.join(PROJECT_ROOT, "exports/texture_final.png")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/comparison/texture_verify")

# Sample frames: first / mid / last
SAMPLE_OBJS = [
    "step_2_frame_000000.obj",
    "step_2_frame_009000.obj",
    "step_2_frame_017995.obj",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def reset_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Remove orphan data
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)


def setup_camera_at_mesh(obj):
    """Add a camera positioned relative to the mesh bounding box."""
    import mathutils

    # Compute world-space bounding box center
    bbox_corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    center = sum(bbox_corners, mathutils.Vector()) / 8
    extents = [
        max(c[i] for c in bbox_corners) - min(c[i] for c in bbox_corners)
        for i in range(3)
    ]
    max_extent = max(extents)

    # Place camera above and to the side at 2× max extent distance
    cam_offset = mathutils.Vector((max_extent * 1.5, -max_extent * 1.5, max_extent * 1.2))
    cam_location = center + cam_offset

    bpy.ops.object.camera_add(location=cam_location)
    cam = bpy.context.active_object

    # Point camera at mesh center
    direction = center - cam_location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = cam

    print(f"  Mesh center: {center}, max_extent: {max_extent:.1f}")
    return cam


def setup_lighting():
    """Add multiple lights for clear texture visibility."""
    # Strong key light
    bpy.ops.object.light_add(type="SUN", location=(1.0, -1.0, 2.0))
    sun = bpy.context.active_object
    sun.data.energy = 8.0
    # Fill light from opposite side
    bpy.ops.object.light_add(type="SUN", location=(-1.0, 1.0, 1.0))
    fill = bpy.context.active_object
    fill.data.energy = 4.0
    import mathutils
    fill.rotation_euler = mathutils.Euler((0.5, 0.0, 3.14), "XYZ")


def load_obj_with_texture(obj_path, texture_path):
    """Import OBJ and ensure texture is applied via Cycles material."""
    bpy.ops.wm.obj_import(filepath=obj_path)
    imported = bpy.context.selected_objects
    if not imported:
        print(f"ERROR: No objects imported from {obj_path}")
        return None

    obj = imported[0]
    bpy.context.view_layer.objects.active = obj

    # Ensure texture image is loaded
    if texture_path not in [img.filepath for img in bpy.data.images]:
        tex_image = bpy.data.images.load(texture_path)
    else:
        tex_image = bpy.data.images[os.path.basename(texture_path)]

    # Create or update material with texture node
    if obj.data.materials:
        mat = obj.data.materials[0]
    else:
        mat = bpy.data.materials.new(name="mouse_texture")
        obj.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Principled BSDF + Image Texture + Material Output
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.image = tex_image

    links.new(tex_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    output_node.location = (400, 0)
    bsdf_node.location = (100, 0)
    tex_node.location = (-200, 0)

    return obj


def render_to_file(output_path, resolution=(512, 512)):
    """Set render settings and render to file."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def main():
    print(f"=== Blender texture verification ===")
    print(f"OBJ dir:   {OBJ_DIR}")
    print(f"Texture:   {TEXTURE_PNG}")
    print(f"Output:    {OUTPUT_DIR}")
    print()

    if not os.path.exists(TEXTURE_PNG):
        print(f"FATAL: texture not found: {TEXTURE_PNG}")
        sys.exit(1)

    for obj_filename in SAMPLE_OBJS:
        obj_path = os.path.join(OBJ_DIR, obj_filename)
        if not os.path.exists(obj_path):
            print(f"SKIP: {obj_filename} not found")
            continue

        stem = os.path.splitext(obj_filename)[0]
        out_png = os.path.join(OUTPUT_DIR, f"{stem}_textured.png")

        print(f"Rendering: {obj_filename} → {out_png}")
        reset_scene()
        setup_lighting()
        obj = load_obj_with_texture(obj_path, TEXTURE_PNG)
        if obj is None:
            continue
        setup_camera_at_mesh(obj)

        render_to_file(out_png)
        size_kb = os.path.getsize(out_png) / 1024
        print(f"  Done: {size_kb:.0f}KB")

    print()
    print(f"=== Verification complete. Check {OUTPUT_DIR} ===")


main()
