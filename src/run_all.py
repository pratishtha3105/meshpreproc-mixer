import os
import trimesh
import json
import numpy as np
from tqdm import tqdm
from load import load_mesh
from normalize import min_max_normalize, unit_sphere_normalize
from quantize import quantize, dequantize
from reconstruct import denormalize_minmax, denormalize_unit_sphere
from metrics import compute_errors, plot_error_bars


# ------------------ MAIN PIPELINE ------------------

def process_mesh(mesh_path, output_dir="outputs", n_bins=1024):
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    print(f"\n🔹 Processing mesh: {mesh_name}")

    # Create subfolders
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Load mesh
    mesh, vertices = load_mesh(mesh_path)

    # --- Normalization ---
    minmax_norm, minmax_meta = min_max_normalize(vertices)
    unit_norm, unit_meta = unit_sphere_normalize(vertices)

    # Save metadata for reproducibility
    with open(os.path.join(output_dir, f"{mesh_name}_minmax_meta.json"), "w") as f:
        json.dump(minmax_meta, f)
    with open(os.path.join(output_dir, f"{mesh_name}_unit_meta.json"), "w") as f:
        json.dump(unit_meta, f)

    # --- Quantization ---
    minmax_q = quantize(minmax_norm, n_bins)
    unit_q = quantize(unit_norm, n_bins)

    # --- Dequantization ---
    minmax_dq = dequantize(minmax_q, n_bins)
    unit_dq = dequantize(unit_q, n_bins)

    # --- Reconstruction ---
    recon_minmax = denormalize_minmax(minmax_dq, minmax_meta)
    recon_unit = denormalize_unit_sphere(unit_dq, unit_meta)

    # --- Error Computation ---
    err_minmax = compute_errors(vertices, recon_minmax)
    err_unit = compute_errors(vertices, recon_unit)

    # --- Save Results ---
    # Save as .npy for data reference
    np.save(os.path.join(output_dir, f"{mesh_name}_recon_minmax.npy"), recon_minmax)
    np.save(os.path.join(output_dir, f"{mesh_name}_recon_unit.npy"), recon_unit)

    # Save as .obj for visualization
    save_obj_minmax = os.path.join(output_dir, f"{mesh_name}_recon_minmax.obj")
    save_obj_unit = os.path.join(output_dir, f"{mesh_name}_recon_unitsphere.obj")

    trimesh.Trimesh(vertices=recon_minmax, faces=mesh.faces, process=False).export(save_obj_minmax)
    trimesh.Trimesh(vertices=recon_unit, faces=mesh.faces, process=False).export(save_obj_unit)

    plot_error_bars(err_minmax, f"{mesh_name} - MinMax", os.path.join(output_dir, "plots", f"{mesh_name}_error_minmax.png"))
    plot_error_bars(err_unit, f"{mesh_name} - UnitSphere", os.path.join(output_dir, "plots", f"{mesh_name}_error_unitsphere.png"))

    print(f"✅ Done: {mesh_name}")
    print(f"   MSE-MinMax: {err_minmax['mse']:.6f}, MSE-UnitSphere: {err_unit['mse']:.6f}")

    return {
        "mesh": mesh_name,
        "vertices": vertices.shape[0],
        "bins": n_bins,
        "mse_minmax": float(err_minmax["mse"]),
        "mae_minmax": float(err_minmax["mae"]),
        "mse_unitsphere": float(err_unit["mse"]),
        "mae_unitsphere": float(err_unit["mae"])
    }


if __name__ == "__main__":
    data_dir = "data"
    output_dir = "outputs"
    n_bins = 1024

    summary = []

    # Find all .obj files
    meshes = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".obj")]
    if not meshes:
        print("❌ No .obj files found in 'data/' folder!")
        exit()

    # Process each mesh
    for mesh_path in tqdm(meshes, desc="Processing all meshes"):
        result = process_mesh(mesh_path, output_dir, n_bins)
        summary.append(result)

    # Save summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    import pandas as pd
    df = pd.DataFrame(summary)
    df.to_csv(summary_path, index=False)
    print(f"\n📊 Summary saved to {summary_path}")
