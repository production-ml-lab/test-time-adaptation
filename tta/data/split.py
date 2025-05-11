import torch
import torch.nn.functional as F


def compute_entropy(probs):
    """Compute entropy for given probabilities."""
    log_probs = torch.log(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs, dim=1)  # entropy per sample
    return entropy


def compute_gradient_norm(entropy, input_sample):
    """Compute gradient norm with respect to input."""
    pseudo_loss = entropy.sum()
    pseudo_loss.backward(retain_graph=True)
    grad = input_sample.grad.detach()
    grad_norm = torch.norm(grad, p=2)  # L2 norm
    return grad_norm


def split_dataset_by_quadrants(
    model,
    dataloader,
    gradient_norm_threshold=20.0,
    entropy_threshold=0.5,
    device="cuda",
):
    """
    Split dataset into 4 quadrants based on gradient norm and entropy thresholds.

    Quadrants:
    1: High Gradient Norm (≥threshold) & High Entropy (≥threshold)
    2: Low Gradient Norm (<threshold) & High Entropy (≥threshold)
    3: Low Gradient Norm (<threshold) & Low Entropy (<threshold)
    4: High Gradient Norm (≥threshold) & Low Entropy (<threshold)

    Args:
        model: PyTorch model
        dataloader: DataLoader containing the dataset
        gradient_norm_threshold: Threshold for gradient norm (default: 20.0)
        entropy_threshold: Threshold for entropy (default: 0.5)
        device: Device to run computations on (default: "cuda")

    Returns:
        quadrant_indices: Dictionary containing indices for each quadrant
        values: Dictionary containing (gradient_norm, entropy) pairs for each sample
    """
    model.eval()
    quadrant_indices = {1: [], 2: [], 3: [], 4: []}
    values = {}

    for idx, batch in enumerate(dataloader):
        input_sample, _, _ = batch
        input_sample = input_sample.to(device).requires_grad_(True)

        # Forward pass
        output = model(input_sample)
        probs = F.softmax(output, dim=1)

        # Compute entropy
        entropy = compute_entropy(probs)
        entropy_val = entropy.item()

        # Compute gradient norm
        grad_norm = compute_gradient_norm(entropy, input_sample)
        grad_norm_val = grad_norm.item()

        # Store values
        values[idx] = (grad_norm_val, entropy_val)

        # Categorize into quadrants
        if grad_norm_val >= gradient_norm_threshold:
            if entropy_val >= entropy_threshold:
                quadrant_indices[1].append(idx)  # Q1: High GN, High E
            else:
                quadrant_indices[4].append(idx)  # Q4: High GN, Low E
        else:
            if entropy_val >= entropy_threshold:
                quadrant_indices[2].append(idx)  # Q2: Low GN, High E
            else:
                quadrant_indices[3].append(idx)  # Q3: Low GN, Low E

        # Clean up
        model.zero_grad()

    return quadrant_indices, values
