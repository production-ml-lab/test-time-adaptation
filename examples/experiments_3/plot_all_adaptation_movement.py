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


def plot_all_movements(
    before_values,
    after_values,
    quadrant_indices,
    gradient_norm_threshold,
    entropy_threshold,
):
    """Plot all data points and their movements with arrows."""
    colors = ["red", "blue", "green", "purple"]
    labels = [
        "Q1: High GN, High E",
        "Q2: Low GN, High E",
        "Q3: Low GN, Low E",
        "Q4: High GN, Low E",
    ]

    plt.figure(figsize=(15, 10))

    # Plot points and arrows for each quadrant
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if not indices:
            continue

        # Get before and after points for this quadrant
        x_before = [before_values[i][0] for i in indices]
        y_before = [before_values[i][1] for i in indices]
        x_after = [after_values[i][0] for i in indices]
        y_after = [after_values[i][1] for i in indices]

        # Plot before points
        plt.scatter(
            x_before,
            y_before,
            c=colors[q - 1],
            label=f"{labels[q-1]} (Before)",
            alpha=0.6,
            marker="o",
        )
        # Plot after points
        plt.scatter(x_after, y_after, c=colors[q - 1], alpha=0.3, marker="^")

        # Plot arrows
        for i in range(len(indices)):
            plt.arrow(
                x_before[i],
                y_before[i],
                x_after[i] - x_before[i],
                y_after[i] - y_before[i],
                color=colors[q - 1],
                alpha=0.2,
                head_width=0.1,
                head_length=0.1,
                length_includes_head=True,
            )

    # Add quadrant lines
    plt.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    plt.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)

    # Add labels and title
    plt.xlabel("Gradient Norm")
    plt.ylabel("Entropy")
    plt.title("Adaptation Movement of All Samples")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save plot
    plt.savefig("adaptation_movement_all_samples.png")
    plt.close()


def main():
    # Initialize model and device
    method = Tent(optim_steps=1)
    model = method.model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)

    # Create dataset and dataloader
    dataset = Cifar10CDataset(
        num_samples=100, corrupt_domain_orders=["gaussian_noise"], severity=5
    )
    dataloader = build_test_loader(dataset, batch_size=1)

    # Split dataset based on quadrants and collect before values
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
    for batch_idx, batch in enumerate(dataloader):
        # Run adaptation
        input_sample, _, _ = batch
        input_sample = input_sample.to(device)
        method.forward_and_adapt(input_sample)

        # Get values after adaptation
        grad_norm_val, entropy_val = get_values_for_batch(model, batch, device)
        after_values[batch_idx] = (grad_norm_val, entropy_val)

    # Plot all movements
    plot_all_movements(
        before_values,
        after_values,
        quadrant_indices,
        gradient_norm_threshold,
        entropy_threshold,
    )
    print("\nPlot saved as 'adaptation_movement_all_samples.png'")

    # Print statistics for each quadrant
    print("\nMovement analysis per quadrant:")
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if indices:
            print(f"\nQuadrant {q}:")
            avg_gn_before = np.mean([before_values[i][0] for i in indices])
            avg_gn_after = np.mean([after_values[i][0] for i in indices])
            avg_e_before = np.mean([before_values[i][1] for i in indices])
            avg_e_after = np.mean([after_values[i][1] for i in indices])

            print(f"Average Gradient Norm: {avg_gn_before:.2f} -> {avg_gn_after:.2f}")
            print(f"Average Entropy: {avg_e_before:.2f} -> {avg_e_after:.2f}")


if __name__ == "__main__":
    main()
