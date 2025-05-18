import matplotlib.pyplot as plt
from pathlib import Path

from tta.method import Tent
from tta.data.dataloader import build_test_loader

from split_dataset import QuadrantConfig, QuadrantSplitter

config = QuadrantConfig()

# Initialize model and splitter
method = Tent(optim_steps=1)
splitter = QuadrantSplitter(method.model, config)
splitter.create_dataset()
before_quadrant_indices, before_values = splitter.split()
splitter.plot_quadrants("./results/before_adapt.png")

# Adapt samples from all quadrants
for quadrant in range(1, 5):
    x, y, label = splitter.dataset[splitter.get_quadrant_samples(quadrant)]
    loader = build_test_loader(x, batch_size=1)
    for i in loader:
        p = method.forward_and_adapt(i)

    # Create new splitter after adaptation
    after_splitter = QuadrantSplitter(method.model, config)
    after_splitter.create_dataset()
    after_quadrant_indices, after_values = after_splitter.split()

    # Plot the results
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
        indices = before_quadrant_indices[q]
        if indices:
            x = [after_values[i][0] for i in indices]  # gradient norms
            y = [after_values[i][1] for i in indices]  # entropy values
            plt.scatter(x, y, c=colors[q - 1], label=labels[q - 1], alpha=0.6)

    # Add quadrant lines
    plt.axvline(
        x=config.gradient_norm_threshold, color="black", linestyle="--", alpha=0.3
    )
    plt.axhline(y=config.entropy_threshold, color="black", linestyle="--", alpha=0.3)

    plt.xlabel("Gradient Norm")
    plt.ylabel("Entropy")
    plt.title(
        f"Sample Distribution in Quadrants\nCorruption: {after_splitter.config.corruption}, Severity: {after_splitter.config.severity}\nAfter Adapting Quadrant {quadrant}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    output_path = f"./results/after_quadrant_{quadrant}.png"
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    method.reset()
