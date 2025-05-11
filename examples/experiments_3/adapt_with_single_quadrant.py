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


def plot_quadrant_effects(
    before_values,
    after_values,
    quadrant_indices,
    adaptation_quadrant,
    gradient_norm_threshold,
    entropy_threshold,
):
    """Plot how adaptation with one quadrant affects all quadrants."""
    colors = ["red", "blue", "green", "purple"]
    labels = [
        "Q1: High GN, High E",
        "Q2: Low GN, High E",
        "Q3: Low GN, Low E",
        "Q4: High GN, Low E",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Before adaptation plot
    ax1.set_title(f"Before Adaptation (Using Q{adaptation_quadrant} for Training)")
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if not indices:
            continue

        x = [before_values[i][0] for i in indices]
        y = [before_values[i][1] for i in indices]
        ax1.scatter(x, y, c=colors[q - 1], label=labels[q - 1], alpha=0.6)

    ax1.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    ax1.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Gradient Norm")
    ax1.set_ylabel("Entropy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # After adaptation plot
    ax2.set_title(f"After Adaptation (Using Q{adaptation_quadrant} for Training)")
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if not indices:
            continue

        x = [after_values[i][0] for i in indices]
        y = [after_values[i][1] for i in indices]
        ax2.scatter(x, y, c=colors[q - 1], label=labels[q - 1], alpha=0.6)

        # Add arrows for the adaptation quadrant
        if q == adaptation_quadrant:
            for i, idx in enumerate(indices):
                ax2.arrow(
                    before_values[idx][0],
                    before_values[idx][1],
                    after_values[idx][0] - before_values[idx][0],
                    after_values[idx][1] - before_values[idx][1],
                    color=colors[q - 1],
                    alpha=0.2,
                    head_width=0.1,
                    head_length=0.1,
                    length_includes_head=True,
                )

    ax2.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    ax2.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Gradient Norm")
    ax2.set_ylabel("Entropy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.savefig(f"adaptation_effects_using_q{adaptation_quadrant}.png")
    plt.close()


def run_experiment_with_quadrant(adaptation_quadrant):
    print(f"\n=== Running adaptation using Quadrant {adaptation_quadrant} ===")

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
    quadrant_indices, before_values = split_dataset_by_quadrants(
        model=model,
        dataloader=full_dataloader,
        gradient_norm_threshold=gradient_norm_threshold,
        entropy_threshold=entropy_threshold,
        device=device,
    )

    # Print initial distribution
    print("\nInitial distribution:")
    for q in range(1, 5):
        print(f"Quadrant {q}: {len(quadrant_indices[q])} samples")

    # Create adaptation dataset (only samples from the chosen quadrant)
    adaptation_indices = quadrant_indices[adaptation_quadrant]
    if len(adaptation_indices) == 0:
        print(f"No samples in Quadrant {adaptation_quadrant}. Skipping experiment.")
        return

    adaptation_dataset = Subset(dataset, adaptation_indices)
    adaptation_dataloader = build_test_loader(adaptation_dataset, batch_size=1)

    # Run adaptation using only the chosen quadrant
    print(
        f"\nAdapting model using {len(adaptation_indices)} samples from Quadrant {adaptation_quadrant}"
    )
    for batch in adaptation_dataloader:
        input_sample, _, _ = batch
        input_sample = input_sample.to(device)
        method.forward_and_adapt(input_sample)

    # Collect after-adaptation values for ALL samples
    after_values = {}
    for batch_idx, batch in enumerate(full_dataloader):
        grad_norm_val, entropy_val = get_values_for_batch(model, batch, device)
        after_values[batch_idx] = (grad_norm_val, entropy_val)

    # Plot results
    plot_quadrant_effects(
        before_values,
        after_values,
        quadrant_indices,
        adaptation_quadrant,
        gradient_norm_threshold,
        entropy_threshold,
    )
    print(f"\nPlot saved as 'adaptation_effects_using_q{adaptation_quadrant}.png'")

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


def main():
    # Run experiments using each quadrant for adaptation
    for adaptation_quadrant in range(1, 5):
        run_experiment_with_quadrant(adaptation_quadrant)


if __name__ == "__main__":
    main()
