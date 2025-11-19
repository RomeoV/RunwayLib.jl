# /// script
# dependencies = ["trimesh", "numpy", "fast-simplification"]
# ///
import trimesh

# Load the mesh
assetname = "A320NeoV2.stl"
mesh = trimesh.load(assetname)

print(f"Original: {len(mesh.faces)} faces")

# Simplify to target face count
target_faces = 500  # Adjust as needed
mesh_simple = mesh.simplify_quadric_decimation(0.9)

print(f"Simplified: {len(mesh_simple.faces)} faces")

# Export
mesh_simple.export(assetname.split('.')[0] + "_lowpoly.stl")
