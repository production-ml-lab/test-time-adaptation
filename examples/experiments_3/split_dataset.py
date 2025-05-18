import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from tta.data.cifar import Cifar10CDataset
from tta.data.dataloader import build_test_loader
from tta.method.source import Source
from tta.data.split import split_dataset_by_quadrants


@dataclass
class QuadrantConfig:
    """Configuration for quadrant-based dataset splitting."""

    gradient_norm_threshold: float = 20.0
    entropy_threshold: float = 0.5
    num_samples: int = 100
    corruption: str = "gaussian_noise"
    severity: int = 5
    batch_size: int = 1


class QuadrantSplitter:
    """
    A class for splitting datasets into quadrants based on gradient norm and entropy.

    This class provides functionality to:
    1. Split a dataset into quadrants based on gradient norm and entropy thresholds
    2. Get samples for specific quadrants
    3. Visualize the quadrant distribution
    4. Access quadrant statistics
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[QuadrantConfig] = None,
        device: Optional[str] = "mps",
    ):
        """
        Initialize the QuadrantSplitter.

        Args:
            model: PyTorch model to use for computing gradients and entropy
            config: Configuration for quadrant splitting (optional)
            device: Device to run computations on (optional)
        """
        self.model = model
        self.config = config or QuadrantConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Initialize storage for results
        self.quadrant_indices: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
        self.values: Dict[int, Tuple[float, float]] = {}
        self.dataset = None
        self.dataloader = None

    def create_dataset(self, dataset_class=Cifar10CDataset, **dataset_kwargs):
        """
        Create and initialize the dataset.

        Args:
            dataset_class: Dataset class to use
            **dataset_kwargs: Additional arguments to pass to dataset constructor
        """
        dataset_kwargs.setdefault("num_samples", self.config.num_samples)
        dataset_kwargs.setdefault("corrupt_domain_orders", [self.config.corruption])
        dataset_kwargs.setdefault("severity", self.config.severity)

        self.dataset = dataset_class(**dataset_kwargs)
        self.dataloader = build_test_loader(
            self.dataset, batch_size=self.config.batch_size
        )
        return self.dataset

    def split(self) -> Tuple[Dict[int, List[int]], Dict[int, Tuple[float, float]]]:
        """
        Split the dataset into quadrants.

        Returns:
            Tuple containing:
            - Dictionary mapping quadrant numbers to lists of sample indices
            - Dictionary mapping sample indices to (gradient_norm, entropy) pairs
        """
        if self.dataloader is None:
            raise ValueError("Dataset not initialized. Call create_dataset() first.")

        self.quadrant_indices, self.values = split_dataset_by_quadrants(
            model=self.model,
            dataloader=self.dataloader,
            gradient_norm_threshold=self.config.gradient_norm_threshold,
            entropy_threshold=self.config.entropy_threshold,
            device=self.device,
        )
        return self.quadrant_indices, self.values

    def get_quadrant_samples(self, quadrant: int) -> List[int]:
        """
        Get the indices of samples in a specific quadrant.

        Args:
            quadrant: Quadrant number (1-4)

        Returns:
            List of sample indices in the specified quadrant
        """
        if not self.quadrant_indices:
            raise ValueError("Dataset not split. Call split() first.")
        return self.quadrant_indices[quadrant]

    def get_quadrant_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get statistics for each quadrant.

        Returns:
            Dictionary mapping quadrant numbers to dictionaries containing:
            - num_samples: Number of samples in the quadrant
            - avg_gradient_norm: Average gradient norm
            - avg_entropy: Average entropy
        """
        if not self.quadrant_indices or not self.values:
            raise ValueError("Dataset not split. Call split() first.")

        stats = {}
        for q in range(1, 5):
            indices = self.quadrant_indices[q]
            if indices:
                gn_values = [self.values[i][0] for i in indices]
                entropy_values = [self.values[i][1] for i in indices]
                stats[q] = {
                    "num_samples": len(indices),
                    "avg_gradient_norm": sum(gn_values) / len(gn_values),
                    "avg_entropy": sum(entropy_values) / len(entropy_values),
                }
        return stats

    def plot_quadrants(
        self,
        output_path: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot the quadrant distribution.

        Args:
            output_path: Path to save the plot
            title: Optional custom title for the plot
            figsize: Figure size as (width, height)
        """
        if not self.quadrant_indices or not self.values:
            raise ValueError("Dataset not split. Call split() first.")

        colors = ["red", "blue", "green", "purple"]
        labels = [
            "Q1: High GN, High E",
            "Q2: Low GN, High E",
            "Q3: Low GN, Low E",
            "Q4: High GN, Low E",
        ]

        plt.figure(figsize=figsize)

        # Plot each quadrant
        for q in range(1, 5):
            indices = self.quadrant_indices[q]
            if indices:
                x = [self.values[i][0] for i in indices]  # gradient norms
                y = [self.values[i][1] for i in indices]  # entropy values
                plt.scatter(x, y, c=colors[q - 1], label=labels[q - 1], alpha=0.6)

        # Add quadrant lines
        plt.axvline(
            x=self.config.gradient_norm_threshold,
            color="black",
            linestyle="--",
            alpha=0.3,
        )
        plt.axhline(
            y=self.config.entropy_threshold, color="black", linestyle="--", alpha=0.3
        )

        plt.xlabel("Gradient Norm")
        plt.ylabel("Entropy")
        plt.title(
            title
            or f"Sample Distribution in Quadrants\nCorruption: {self.config.corruption}, Severity: {self.config.severity}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path)
        plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split dataset into quadrants based on gradient norm and entropy"
    )

    # Dataset arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to use from the dataset",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default="gaussian_noise",
        help="Type of corruption to apply to the dataset",
    )
    parser.add_argument(
        "--severity", type=int, default=5, help="Severity of corruption (1-5)"
    )

    # Threshold arguments
    parser.add_argument(
        "--gradient-norm-threshold",
        type=float,
        default=20.0,
        help="Threshold for gradient norm",
    )
    parser.add_argument(
        "--entropy-threshold", type=float, default=0.5, help="Threshold for entropy"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="quadrants_plot.png",
        help="Name of the output plot file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model and splitter
    method = Source()
    config = QuadrantConfig(
        gradient_norm_threshold=args.gradient_norm_threshold,
        entropy_threshold=args.entropy_threshold,
        num_samples=args.num_samples,
        corruption=args.corruption,
        severity=args.severity,
    )

    splitter = QuadrantSplitter(method.model, config)

    # Create and split dataset
    splitter.create_dataset()
    quadrant_indices, values = splitter.split()

    # Print results
    print(f"\nTotal samples: {len(splitter.dataset)}")
    stats = splitter.get_quadrant_stats()
    for q in range(1, 5):
        if q in stats:
            print(f"\nQuadrant {q}:")
            print(f"Number of samples: {stats[q]['num_samples']}")
            print(f"Average Gradient Norm: {stats[q]['avg_gradient_norm']:.2f}")
            print(f"Average Entropy: {stats[q]['avg_entropy']:.2f}")

    # Plot the quadrants
    output_path = Path(args.output_dir) / args.plot_name
    splitter.plot_quadrants(str(output_path))
    print(f"\nPlot saved as '{output_path}'")


if __name__ == "__main__":
    main()
