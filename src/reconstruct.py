import numpy as np
import os
import trimesh

# ------------------ DE-NORMALIZATION HELPERS ------------------

def denormalize_minmax(dequantized_vertices, meta):
    """Reverse min-max normalization using original min and max values."""
    vmin = np.array(meta["vmin"])
    vmax = np.array(meta["vmax"])
    reconstructed = dequantized_vertices * (vmax - vmin) + vmin
    return reconstructed


def denormalize_unit_sphere(dequantized_vertices, meta):
    """Reverse unit-sphere normalization using original centroid and scale."""
    centroid = np.array(meta["centroid"])
    scale = meta["scale"]
    reconstructed = dequantized_vertices * scale + centroid
    return reconstructed


# ------------------ SAVE AS .OBJ ------------------

def save_as_obj(vertices, faces, out_path):
    """
    Save mesh as .obj using Trimesh.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(out_path)
    print(f"💾 Saved reconstructed mesh: {out_path}")


# ------------------ TESTING ------------------
if __name__ == "__main__":
    from quantize import dequantize
    from normalize import min_max_normalize, unit_sphere_normalize
    from load import load_mesh

    mesh, vertices = load_mesh("data/girl.obj")
    print("✅ Original vertices loaded:", vertices.shape)

    # Apply normalizations
    minmax_norm, minmax_meta = min_max_normalize(vertices)
    unit_norm, unit_meta = unit_sphere_normalize(vertices)

    # Quantize + Dequantize
    q_minmax = np.floor(minmax_norm * 1023).astype(np.int32)
    dq_minmax = dequantize(q_minmax, 1024)

    q_unit = np.floor(unit_norm * 1023).astype(np.int32)
    dq_unit = dequantize(q_unit, 1024)

    # Reconstruct
    recon_minmax = denormalize_minmax(dq_minmax, minmax_meta)
    recon_unit = denormalize_unit_sphere(dq_unit, unit_meta)

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Save reconstructed meshes in .obj format
    save_as_obj(recon_minmax, mesh.faces, "outputs/girl_recon_minmax.obj")
    save_as_obj(recon_unit, mesh.faces, "outputs/girl_recon_unitsphere.obj")
