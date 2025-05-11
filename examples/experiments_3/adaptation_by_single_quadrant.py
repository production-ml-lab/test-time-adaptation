import torch
import matplotlib.pyplot as plt
import numpy as np
from tta.data.cifar import Cifar10CDataset
from tta.data.dataloader import build_test_loader
from tta.method.tent import Tent
from tta.data.split import (
    split_dataset_by_quadrants,
    compute_entropy,
    compute_gradient_norm,
)
from torch.utils.data import Subset


def plot_quadrant_adaptation(
    before_values, after_values, quadrant, gradient_norm_threshold, entropy_threshold
):
    """Plot before and after adaptation results for a single quadrant."""
    colors = ["red", "blue", "green", "purple"]
    labels = [
        "Q1: High GN, High E",
        "Q2: Low GN, High E",
        "Q3: Low GN, Low E",
        "Q4: High GN, Low E",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Before adaptation plot
    ax1.set_title(f"Before Adaptation - {labels[quadrant-1]}")
    x = [val[0] for val in before_values]  # gradient norms
    y = [val[1] for val in before_values]  # entropy values
    ax1.scatter(x, y, c=colors[quadrant - 1], label=labels[quadrant - 1], alpha=0.6)

    ax1.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    ax1.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Gradient Norm")
    ax1.set_ylabel("Entropy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # After adaptation plot
    ax2.set_title(f"After Adaptation - {labels[quadrant-1]}")
    x = [val[0] for val in after_values]  # gradient norms
    y = [val[1] for val in after_values]  # entropy values
    ax2.scatter(x, y, c=colors[quadrant - 1], label=labels[quadrant - 1], alpha=0.6)

    ax2.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    ax2.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Gradient Norm")
    ax2.set_ylabel("Entropy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.savefig(f"adaptation_results_quadrant_{quadrant}.png")
    plt.close()


def get_values_for_batch(model, batch, device):
    """Compute gradient norm and entropy values for a batch."""
    input_sample, _, _ = batch
    input_sample = input_sample.to(device).requires_grad_(True)

    # Forward pass
    output = model(input_sample)
    probs = torch.softmax(output, dim=1)

    # Compute entropy
    entropy = compute_entropy(probs)
    entropy_val = entropy.item()

    # Compute gradient norm
    grad_norm = compute_gradient_norm(entropy, input_sample)
    grad_norm_val = grad_norm.item()

    model.zero_grad()
    return grad_norm_val, entropy_val


def run_experiment_for_quadrant(quadrant_number):
    print(f"\n=== Running experiment for Quadrant {quadrant_number} ===")

    # Initialize model and device
    method = Tent(optim_steps=1)
    model = method.model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)

    # Create dataset and dataloader
    dataset = Cifar10CDataset(
        num_samples=100, corrupt_domain_orders=["gaussian_noise"], severity=5
    )
    full_dataloader = build_test_loader(dataset, batch_size=1)

    # Split dataset based on quadrants
    gradient_norm_threshold = 20.0
    entropy_threshold = 0.5
    quadrant_indices, _ = split_dataset_by_quadrants(
        model=model,
        dataloader=full_dataloader,
        gradient_norm_threshold=gradient_norm_threshold,
        entropy_threshold=entropy_threshold,
        device=device,
    )

    # Get indices for the current quadrant
    current_quadrant_indices = quadrant_indices[quadrant_number]
    print(
        f"Number of samples in Quadrant {quadrant_number}: {len(current_quadrant_indices)}"
    )

    if len(current_quadrant_indices) == 0:
        print("No samples in this quadrant. Skipping experiment.")
        return

    # Create a subset dataset for the current quadrant
    quadrant_dataset = Subset(dataset, current_quadrant_indices)
    quadrant_dataloader = build_test_loader(quadrant_dataset, batch_size=1)

    # Collect before adaptation values
    before_values = []
    model.eval()
    for batch in quadrant_dataloader:
        grad_norm_val, entropy_val = get_values_for_batch(model, batch, device)
        before_values.append((grad_norm_val, entropy_val))

    # Run adaptation and collect after values
    after_values = []
    for batch in quadrant_dataloader:
        # Run adaptation
        input_sample, _, _ = batch
        input_sample = input_sample.to(device)
        method.forward_and_adapt(input_sample)

        # Get values after adaptation
        grad_norm_val, entropy_val = get_values_for_batch(model, batch, device)
        after_values.append((grad_norm_val, entropy_val))

    # Calculate and print statistics
    avg_gn_before = np.mean([v[0] for v in before_values])
    avg_gn_after = np.mean([v[0] for v in after_values])
    avg_e_before = np.mean([v[1] for v in before_values])
    avg_e_after = np.mean([v[1] for v in after_values])

    print(f"\nQuadrant {quadrant_number} Statistics:")
    print(f"Average Gradient Norm: {avg_gn_before:.2f} -> {avg_gn_after:.2f}")
    print(f"Average Entropy: {avg_e_before:.2f} -> {avg_e_after:.2f}")

    # Plot results
    plot_quadrant_adaptation(
        before_values,
        after_values,
        quadrant_number,
        gradient_norm_threshold,
        entropy_threshold,
    )
    print(f"Plot saved as 'adaptation_results_quadrant_{quadrant_number}.png'")


def main():
    # Run experiments for each quadrant
    for quadrant in range(1, 5):
        run_experiment_for_quadrant(quadrant)


if __name__ == "__main__":
    main()
