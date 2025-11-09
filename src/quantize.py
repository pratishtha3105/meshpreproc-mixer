import numpy as np
import os

# ------------------ QUANTIZATION ------------------
def quantize(normalized_vertices, n_bins=1024):
    """
    Quantize normalized vertices into discrete integer bins (0 to n_bins-1).
    Example: n_bins=1024 -> each axis value is in [0, 1023]
    """
    q = np.floor(normalized_vertices * (n_bins - 1)).astype(np.int32)
    q = np.clip(q, 0, n_bins - 1)
    return q


# ------------------ DEQUANTIZATION ------------------
def dequantize(q_vertices, n_bins=1024):
    """
    Convert quantized integer vertices back to float values in [0,1].
    """
    return q_vertices.astype(np.float64) / (n_bins - 1)


# ------------------ TESTING ------------------
if __name__ == "__main__":
    import json

    # Load normalized data from outputs
    minmax_path = "outputs/girl_minmax.npy"
    unitsphere_path = "outputs/girl_unitsphere.npy"
    os.makedirs("outputs", exist_ok=True)

    minmax_norm = np.load(minmax_path)
    unitsphere_norm = np.load(unitsphere_path)

    # Quantize both normalized versions
    n_bins = 1024
    minmax_q = quantize(minmax_norm, n_bins)
    unitsphere_q = quantize(unitsphere_norm, n_bins)

    # Save quantized data
    np.save("outputs/girl_minmax_quantized.npy", minmax_q)
    np.save("outputs/girl_unitsphere_quantized.npy", unitsphere_q)

    print("✅ Quantized both normalization methods.")
    print("Quantized shape:", minmax_q.shape)
    print("📁 Saved quantized files to outputs/")
