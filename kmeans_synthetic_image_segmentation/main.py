import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
from scipy.optimize import linear_sum_assignment
import warnings
import os

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Synthetic Data Generation
# -----------------------------------------------------------------------------
def generate_synthetic_image(size=(200, 200), n_blobs=4, blob_std=15, random_state=42,
                             add_noise=True, noise_level=0.05):
    """
    Generates a synthetic grayscale image composed of multiple Gaussian blobs and
    corresponding ground truth labels.

    Args:
        size (tuple): (height, width) of the image.
        n_blobs (int): Number of Gaussian blobs to generate.
        blob_std (float): Standard deviation of Gaussian blobs.
        random_state (int): Seed for reproducibility.
        add_noise (bool): Whether to add Gaussian noise to the image.
        noise_level (float): Std dev of Gaussian noise added.

    Returns:
        image (np.ndarray): Synthetic grayscale image (float32, 0-1).
        ground_truth (np.ndarray): Integer labels for each pixel indicating blob membership.
        blob_centers (list of tuples): Coordinates of Gaussian blob centers.
    """
    np.random.seed(random_state)
    height, width = size
    image = np.zeros(size, dtype=np.float32)
    ground_truth = np.zeros(size, dtype=np.int32)

    margin = int(blob_std * 3)
    blob_centers = []

    # Generate random centers for Gaussian blobs within margins
    for i in range(n_blobs):
        y = np.random.randint(margin, height - margin)
        x = np.random.randint(margin, width - margin)
        blob_centers.append((y, x))

    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Create each Gaussian blob with random intensity, add to image
    for idx, (cy, cx) in enumerate(blob_centers, start=1):
        intensity = np.random.uniform(0.5, 1.0)
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        blob = intensity * np.exp(-dist_sq / (2 * blob_std ** 2))
        image += blob
        mask = blob > 0.1
        # Assign label to ground truth only where this blob dominates
        ground_truth[mask] = idx

    # Normalize image intensities to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Optionally add Gaussian noise
    if add_noise:
        noise = np.random.normal(0, noise_level, size)
        image += noise
        image = np.clip(image, 0, 1)

    return image, ground_truth, blob_centers


# -----------------------------------------------------------------------------
# Visualization Utilities
# -----------------------------------------------------------------------------
def plot_synthetic_image(image, ground_truth, blob_centers, save_path=None):
    """
    Plots the synthetic grayscale image alongside its ground truth segmentation.

    Args:
        image (np.ndarray): Grayscale image.
        ground_truth (np.ndarray): Integer label map.
        blob_centers (list of tuples): Coordinates of Gaussian blob centers.
        save_path (str or None): Path to save the figure. If None, just shows.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot grayscale image with blob centers marked
    ax = axes[0]
    im = ax.imshow(image, cmap='gray')
    ax.set_title("Synthetic Grayscale Image")
    ax.axis('off')
    for (cy, cx) in blob_centers:
        ax.plot(cx, cy, 'rX', markersize=10, markeredgewidth=2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot ground truth segmentation with labels
    ax = axes[1]
    n_labels = np.max(ground_truth)
    cmap = plt.get_cmap('tab10', n_labels)
    im = ax.imshow(ground_truth, cmap=cmap, vmin=0.5, vmax=n_labels + 0.5)
    ax.set_title("Ground Truth Segmentation")
    ax.axis('off')
    for (cy, cx) in blob_centers:
        ax.plot(cx, cy, 'rX', markersize=10, markeredgewidth=2)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(1, n_labels + 1))
    cbar.ax.set_ylabel("Blob Label")

    plt.suptitle("Synthetic Image and Ground Truth", fontsize=18)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_clustered_image(image, labels, n_clusters, save_path=None):
    """
    Plots the original grayscale image alongside the clustering segmentation result.

    Args:
        image (np.ndarray): Grayscale image.
        labels (np.ndarray): Cluster labels per pixel (flattened).
        n_clusters (int): Number of clusters.
        save_path (str or None): Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original grayscale image
    ax = axes[0]
    im = ax.imshow(image, cmap='gray')
    ax.set_title("Original Image")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Cluster segmentation colored
    ax = axes[1]
    cmap = plt.get_cmap('tab10', n_clusters)
    im = ax.imshow(labels.reshape(image.shape), cmap=cmap, vmin=0, vmax=n_clusters - 1)
    ax.set_title(f"K-Means Clustering (K={n_clusters})")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_clusters))
    cbar.ax.set_ylabel("Cluster ID")

    plt.suptitle("Image Segmentation via K-Means", fontsize=18)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_cluster_boundaries(image, labels, title="Cluster Boundaries Overlay", save_path=None):
    """
    Overlays cluster boundaries detected in the label image onto the grayscale image.

    Args:
        image (np.ndarray): Grayscale image.
        labels (np.ndarray): Cluster labels (flattened or image-shaped).
        title (str): Plot title.
        save_path (str or None): Path to save the figure.
    """
    if labels.ndim == 1:
        labels_img = labels.reshape(image.shape)
    else:
        labels_img = labels

    boundaries = find_boundaries(labels_img, mode='outer')
    overlay = np.stack([image] * 3, axis=-1)
    overlay[boundaries, :] = [1, 0, 0]  # red boundaries

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_metric_vs_k(results, metric_name='Silhouette Score', save_path=None):
    """
    Plots a clustering metric (e.g., silhouette score or inertia) against number of clusters K.

    Args:
        results (list of dict): Each dict has keys 'k', 'silhouette', 'inertia'.
        metric_name (str): Metric to plot ('Silhouette Score' or 'Inertia').
        save_path (str or None): Path to save figure.
    """
    ks = [r['k'] for r in results]
    if metric_name.lower().startswith('silhouette'):
        values = [r['silhouette'] for r in results]
        ylabel = "Silhouette Score (higher is better)"
    elif metric_name.lower() == 'inertia':
        values = [r['inertia'] for r in results]
        ylabel = "Inertia (lower is better)"
    else:
        raise ValueError("Unsupported metric for plotting")

    plt.figure(figsize=(10, 6))
    plt.plot(ks, values, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} vs K for K-Means Clustering")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------------------------------
# Clustering Functions
# -----------------------------------------------------------------------------
def run_kmeans_on_image(image, n_clusters=4, random_state=42):
    """
    Runs K-Means clustering on the pixel intensities of an image.

    Args:
        image (np.ndarray): 2D grayscale image normalized to [0,1].
        n_clusters (int): Number of clusters for K-Means.
        random_state (int): Random seed.

    Returns:
        labels (np.ndarray): Cluster labels per pixel (flattened).
        kmeans_model (KMeans): Fitted KMeans object.
    """
    pixels = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15, max_iter=500)
    labels = kmeans.fit_predict(pixels)
    return labels, kmeans


def evaluate_clustering(image, labels, kmeans_model):
    """
    Computes clustering quality metrics.

    Args:
        image (np.ndarray): Original image.
        labels (np.ndarray): Cluster labels (flattened).
        kmeans_model (KMeans): Fitted KMeans object.

    Returns:
        silhouette (float): Silhouette score (higher better).
        inertia (float): Sum of squared distances within clusters (lower better).
    """
    pixels = image.flatten().reshape(-1, 1)
    try:
        silhouette = silhouette_score(pixels, labels)
    except:
        silhouette = -1  # Cannot compute for trivial cases
    inertia = kmeans_model.inertia_
    return silhouette, inertia


def tune_k_for_image(image, k_range=(2, 10), random_state=42, save_dir=None):
    """
    Runs K-Means clustering for a range of K values and evaluates them.

    Args:
        image (np.ndarray): Grayscale image.
        k_range (tuple): (min_k, max_k) inclusive.
        random_state (int): Random seed.
        save_dir (str or None): Directory to save metric plots.

    Returns:
        best_k (int): Number of clusters with highest silhouette score.
        results (list of dict): Each dict contains 'k', 'silhouette', and 'inertia'.
    """
    results = []
    for k in range(k_range[0], k_range[1] + 1):
        labels, model = run_kmeans_on_image(image, n_clusters=k, random_state=random_state)
        silhouette, inertia = evaluate_clustering(image, labels, model)
        results.append({'k': k, 'silhouette': silhouette, 'inertia': inertia})
        print(f"K={k}: Silhouette={silhouette:.4f}, Inertia={inertia:.2f}")

    best = max(results, key=lambda x: x['silhouette'])
    best_k = best['k']
    print(f"\nBest K found: {best_k} with silhouette score {best['silhouette']:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_metric_vs_k(results, metric_name='Silhouette Score', save_path=os.path.join(save_dir, 'silhouette_vs_k.png'))
        plot_metric_vs_k(results, metric_name='Inertia', save_path=os.path.join(save_dir, 'inertia_vs_k.png'))

    return best_k, results


def compute_accuracy(ground_truth, predicted_labels):
    """
    Computes approximate accuracy of segmentation by matching predicted cluster labels
    to ground truth labels using Hungarian algorithm.

    Args:
        ground_truth (np.ndarray): 2D ground truth labels.
        predicted_labels (np.ndarray): Flattened predicted cluster labels.

    Returns:
        accuracy (float): Pixel-wise accuracy (0-1).
    """
    gt = ground_truth.flatten()
    pred = predicted_labels
    n_clusters = np.max(gt)
    n_pred_clusters = np.max(pred) + 1

    # Confusion matrix between GT and predicted clusters
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt, pred, labels=np.arange(n_clusters+1))

    # Hungarian matching to find best cluster-label alignment
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))

    # Remap predicted labels to ground truth labels
    mapped_preds = np.zeros_like(pred)
    for pred_label, gt_label in mapping.items():
        mapped_preds[pred == pred_label] = gt_label

    accuracy = np.mean(mapped_preds == gt)
    return accuracy


# -----------------------------------------------------------------------------
# Additional Visualization Helpers
# -----------------------------------------------------------------------------
def plot_cluster_centers(kmeans_model, save_path=None):
    """
    Plots cluster centers as a bar chart.

    Args:
        kmeans_model (KMeans): Trained KMeans object.
        save_path (str or None): Path to save figure.
    """
    centers = kmeans_model.cluster_centers_.flatten()
    n_clusters = len(centers)

    plt.figure(figsize=(8, 5))
    plt.bar(range(n_clusters), centers, color='skyblue')
    plt.xlabel('Cluster ID')
    plt.ylabel('Cluster Center Intensity')
    plt.title('K-Means Cluster Centers')
    plt.xticks(range(n_clusters))
    plt.grid(axis='y')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
