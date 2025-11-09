import trimesh
import numpy as np
import os

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    vertices = np.asarray(mesh.vertices)
    return mesh, vertices

def save_stats(mesh_name, vertices, out_dir):
    stats = {
        "num_vertices": vertices.shape[0],
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist(),
        "mean": vertices.mean(axis=0).tolist(),
        "std": vertices.std(axis=0).tolist()
    }
    np.save(os.path.join(out_dir, f"{mesh_name}_stats.npy"), stats)
    print(f"✅ Stats saved for {mesh_name}")

if __name__ == "__main__":
    mesh_path = "data/girl.obj"

    mesh, vertices = load_mesh(mesh_path)
    print("Vertices shape:", vertices.shape)
