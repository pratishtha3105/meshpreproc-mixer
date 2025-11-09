import numpy as np

# ------------------ MIN-MAX NORMALIZATION ------------------
def min_max_normalize(vertices):
    """
    Normalize vertices to the [0, 1] range along each axis.
    """
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    denom = (vmax - vmin)
    denom[denom == 0] = 1.0  # avoid division by zero
    normalized = (vertices - vmin) / denom

    meta = {
        "method": "minmax",
        "vmin": vmin.tolist(),
        "vmax": vmax.tolist()
    }
    return normalized, meta


# ------------------ UNIT SPHERE NORMALIZATION ------------------
def unit_sphere_normalize(vertices):
    """
    Normalize vertices so that the model fits inside a unit sphere centered at origin.
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    max_dist = np.linalg.norm(centered, axis=1).max()
    if max_dist == 0:
        max_dist = 1.0
    normalized = centered / max_dist

    meta = {
        "method": "unit_sphere",
        "centroid": centroid.tolist(),
        "scale": float(max_dist)
    }
    return normalized, meta


# ------------------ TESTING ------------------
if __name__ == "__main__":
    from load import load_mesh
    import os

    mesh_path = "data/girl.obj"
    mesh, vertices = load_mesh(mesh_path)

    # Apply Min-Max Normalization
    minmax_norm, minmax_meta = min_max_normalize(vertices)
    print("✅ Min-Max Normalized vertices shape:", minmax_norm.shape)

    # Apply Unit Sphere Normalization
    sphere_norm, sphere_meta = unit_sphere_normalize(vertices)
    print("✅ Unit Sphere Normalized vertices shape:", sphere_norm.shape)

    # Save normalized results
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/girl_minmax.npy", minmax_norm)
    np.save("outputs/girl_unitsphere.npy", sphere_norm)
    print("📁 Saved normalized meshes to outputs/")
