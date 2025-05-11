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


def plot_adaptation_results(
    before_values,
    after_values,
    quadrant_indices,
    gradient_norm_threshold,
    entropy_threshold,
):
    """Plot before and after adaptation results for each quadrant."""
    colors = ["red", "blue", "green", "purple"]
    labels = [
        "Q1: High GN, High E",
        "Q2: Low GN, High E",
        "Q3: Low GN, Low E",
        "Q4: High GN, Low E",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Before adaptation plot
    ax1.set_title("Before Adaptation")
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if indices:
            x = [before_values[i][0] for i in indices]  # gradient norms
            y = [before_values[i][1] for i in indices]  # entropy values
            ax1.scatter(x, y, c=colors[q - 1], label=labels[q - 1], alpha=0.6)

    ax1.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    ax1.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Gradient Norm")
    ax1.set_ylabel("Entropy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # After adaptation plot
    ax2.set_title("After Adaptation")
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if indices:
            x = [after_values[i][0] for i in indices]  # gradient norms
            y = [after_values[i][1] for i in indices]  # entropy values
            ax2.scatter(x, y, c=colors[q - 1], label=labels[q - 1], alpha=0.6)

    ax2.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    ax2.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Gradient Norm")
    ax2.set_ylabel("Entropy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.savefig("adaptation_results_tent.png")
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


def main():
    # Initialize model and device
    method = Tent(optim_steps=1)  # Using 1 optimization step for each adaptation
    model = method.model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)

    # Create dataset and dataloader
    dataset = Cifar10CDataset(
        num_samples=100, corrupt_domain_orders=["gaussian_noise"], severity=5
    )
    dataloader = build_test_loader(dataset, batch_size=1)

    # Split dataset based on quadrants
    gradient_norm_threshold = 20.0
    entropy_threshold = 0.5
    quadrant_indices, before_values = split_dataset_by_quadrants(
        model=model,
        dataloader=dataloader,
        gradient_norm_threshold=gradient_norm_threshold,
        entropy_threshold=entropy_threshold,
        device=device,
    )

    # Print initial distribution
    print("Initial distribution:")
    for q in range(1, 5):
        print(f"Quadrant {q}: {len(quadrant_indices[q])} samples")

    # Run adaptation and collect after values
    after_values = {}

    # Process each batch
    for batch_idx, batch in enumerate(dataloader):
        # Run adaptation
        input_sample, _, _ = batch
        input_sample = input_sample.to(device)
        method.forward_and_adapt(input_sample)

        # Get values after adaptation
        grad_norm_val, entropy_val = get_values_for_batch(model, batch, device)
        after_values[batch_idx] = (grad_norm_val, entropy_val)

    # Plot results
    plot_adaptation_results(
        before_values,
        after_values,
        quadrant_indices,
        gradient_norm_threshold,
        entropy_threshold,
    )

    # Calculate statistics
    print("\nMovement analysis:")
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if indices:
            print(f"\nQuadrant {q}:")
            # Calculate average movement
            avg_gn_before = np.mean([before_values[i][0] for i in indices])
            avg_gn_after = np.mean([after_values[i][0] for i in indices])
            avg_e_before = np.mean([before_values[i][1] for i in indices])
            avg_e_after = np.mean([after_values[i][1] for i in indices])

            print(f"Average Gradient Norm: {avg_gn_before:.2f} -> {avg_gn_after:.2f}")
            print(f"Average Entropy: {avg_e_before:.2f} -> {avg_e_after:.2f}")

    print("\nPlot saved as 'adaptation_results_tent.png'")


if __name__ == "__main__":
    main()
