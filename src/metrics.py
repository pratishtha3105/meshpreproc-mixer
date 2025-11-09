import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ ERROR COMPUTATION ------------------

def compute_errors(original, reconstructed):
    """
    Compute Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    for each axis and overall.
    """
    diff = original - reconstructed
    mse_axis = np.mean(diff ** 2, axis=0)
    mae_axis = np.mean(np.abs(diff), axis=0)
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    return {
        "mse_axis": mse_axis,
        "mae_axis": mae_axis,
        "mse": mse,
        "mae": mae
    }


def plot_error_bars(errors, title, out_path):
    """
    Plot error per axis and save.
    """
    axes = ["X", "Y", "Z"]
    plt.figure(figsize=(6,4))
    plt.bar(axes, errors["mse_axis"], color='orange')
    plt.title(f"{title} - MSE per axis")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------ TESTING ------------------
if __name__ == "__main__":
    from load import load_mesh

    mesh, vertices = load_mesh("data/girl.obj")
    recon_minmax = np.load("outputs/girl_recon_minmax.npy")
    recon_unitsphere = np.load("outputs/girl_recon_unitsphere.npy")

    os.makedirs("outputs/plots", exist_ok=True)

    # Compute errors
    err_minmax = compute_errors(vertices, recon_minmax)
    err_unit = compute_errors(vertices, recon_unitsphere)

    # Print results
    print("\n🔹 Min-Max Reconstruction Error:")
    print("MSE:", err_minmax["mse"], "| MAE:", err_minmax["mae"])

    print("\n🔹 Unit Sphere Reconstruction Error:")
    print("MSE:", err_unit["mse"], "| MAE:", err_unit["mae"])

    # Plot errors per axis
    plot_error_bars(err_minmax, "Min-Max", "outputs/plots/error_minmax.png")
    plot_error_bars(err_unit, "Unit Sphere", "outputs/plots/error_unitsphere.png")

    print("📊 Error plots saved in outputs/plots/")
