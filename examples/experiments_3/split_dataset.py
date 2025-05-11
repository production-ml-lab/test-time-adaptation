import torch
import matplotlib.pyplot as plt
from tta.data.cifar import Cifar10CDataset
from tta.data.dataloader import build_test_loader
from tta.method.source import Source
from tta.data.split import split_dataset_by_quadrants


def plot_quadrants(
    values, quadrant_indices, gradient_norm_threshold, entropy_threshold
):
    """Plot samples in their respective quadrants."""
    colors = ["red", "blue", "green", "purple"]
    labels = [
        "Q1: High GN, High E",
        "Q2: Low GN, High E",
        "Q3: Low GN, Low E",
        "Q4: High GN, Low E",
    ]

    plt.figure(figsize=(10, 8))

    # Plot each quadrant
    for q in range(1, 5):
        indices = quadrant_indices[q]
        if indices:
            x = [values[i][0] for i in indices]  # gradient norms
            y = [values[i][1] for i in indices]  # entropy values
            plt.scatter(x, y, c=colors[q - 1], label=labels[q - 1], alpha=0.6)

    # Add quadrant lines
    plt.axvline(x=gradient_norm_threshold, color="black", linestyle="--", alpha=0.3)
    plt.axhline(y=entropy_threshold, color="black", linestyle="--", alpha=0.3)

    plt.xlabel("Gradient Norm")
    plt.ylabel("Entropy")
    plt.title("Sample Distribution in Quadrants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("quadrants_plot.png")
    plt.close()


def main():
    # Initialize model
    method = Source()
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
    quadrant_indices, values = split_dataset_by_quadrants(
        model=model,
        dataloader=dataloader,
        gradient_norm_threshold=gradient_norm_threshold,
        entropy_threshold=entropy_threshold,
        device=device,
    )

    # Print results
    print(f"Total samples: {len(dataset)}")
    for q in range(1, 5):
        print(f"\nQuadrant {q} samples: {len(quadrant_indices[q])}")
        print(f"Quadrant {q} indices: {quadrant_indices[q]}")

    # Plot the quadrants
    plot_quadrants(values, quadrant_indices, gradient_norm_threshold, entropy_threshold)
    print("\nPlot saved as 'quadrants_plot.png'")


if __name__ == "__main__":
    main()
