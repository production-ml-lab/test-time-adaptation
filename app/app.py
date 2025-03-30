import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.sidebar.title("TTA Demo app")

############ Sidebar section ############
from tta.misc.registry import ADAPTATION_REGISTRY
from tta.config.utils import load_default_config
from tta.misc.utils import cfg_node_to_dict

method_name = st.sidebar.selectbox("Select Method", ["Source", "Tent", "SHOT"])
shift_type = st.sidebar.selectbox(
    "Select Shift Type",
    [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ],
)

shift_severity = st.sidebar.slider(
    "Select Shift Severity",
    min_value=1,
    max_value=5,
    value=5,
)
optim_steps = st.sidebar.number_input(
    "Optimization Steps",
    min_value=1,
    value=10,
    step=1,
)
batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=1,
    value=4,
    step=1,
)
############ Adaptation section ############
adapt_registry = ADAPTATION_REGISTRY

config = load_default_config()
method_kwargs = cfg_node_to_dict(config.METHOD)
model_name = config.MODEL.NAME
model_backend = config.MODEL.BACKEND
model_pretrain = config.MODEL.PRETRAIN
method_kwargs["optim_steps"] = optim_steps


@st.cache_resource()
def get_method(
    method_name,
    model_name,
    model_backend,
    model_pretrain,
    method_kwargs,
):
    return adapt_registry.get(method_name)(
        model_name=model_name,
        model_backend=model_backend,
        model_pretrain=model_pretrain,
        **method_kwargs,
    )


method = get_method(
    method_name,
    model_name,
    model_backend,
    model_pretrain,
    method_kwargs,
)
method.reset()

############ Data Adaptation ############
import torch
import torch.nn.functional as F

from tta.data.cifar import Cifar10CDataset
from tta.data.dataloader import build_test_loader


@st.cache_data()
def load_dataloader(shift_type, shift_severity, batch_size):
    dataset = Cifar10CDataset(
        num_samples=100,
        corrupt_domain_orders=[shift_type],
        severity=shift_severity,
    )
    data_loader = build_test_loader(dataset, batch_size=batch_size)
    return data_loader


dataloader = load_dataloader(
    shift_type=shift_type, shift_severity=shift_severity, batch_size=batch_size
)


def compute_entropy(probs):
    # Compute entropy
    log_probs = torch.log(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs, dim=1)  # entropy per sample
    return entropy


def compute_cross_entropy(output, label):
    cross_entropy_loss = F.cross_entropy(output, label)
    return cross_entropy_loss


def compute_gradient_norm(entropy, input_sample):
    # Use entropy (or max logit) as pseudo-loss to compute gradients
    pseudo_loss = entropy.sum()
    pseudo_loss.backward(retain_graph=True)

    # Compute gradient norm w.r.t. input
    grad = input_sample.grad.detach()
    grad_norm = torch.norm(grad, p=2)  # L2 norm
    return grad_norm


@st.cache_data()
def calculate():
    before_entropy_values = []
    before_gradient_norms = []
    before_cross_entropy_values = []
    after_entropy_values = []
    after_gradient_norms = []
    after_cross_entropy_values = []

    device = "mps"
    for batch in dataloader:  # single sample or batch size = 1
        input_sample, label, _ = batch

        ############ Before adaptation ############
        for i in range(len(input_sample)):
            method.model.eval()
            before_micro_batch = input_sample[[i]].to(device).requires_grad_(True)
            # Forward pass
            output = method.model(before_micro_batch)
            probs = F.softmax(output, dim=1)

            # Compute entropy
            entropy = compute_entropy(probs)
            before_entropy_values.append(entropy.item())

            # Compute cross-entropy loss
            cross_entropy_loss = compute_cross_entropy(output, label[[i]].to(device))
            before_cross_entropy_values.append(cross_entropy_loss.item())

            # Use entropy (or max logit) as pseudo-loss to compute gradients
            pseudo_loss = entropy.sum()
            pseudo_loss.backward(retain_graph=True)

            # Compute gradient norm w.r.t. input
            grad_norm = compute_gradient_norm(entropy, before_micro_batch)
            before_gradient_norms.append(grad_norm.item())

            method.model.zero_grad()
            # print("before", grad_norm.item())

        ############ Adaptation ############
        x = input_sample.to(device)
        method.forward_and_adapt(x)

        ############ After adaptation ############
        for i in range(len(input_sample)):
            method.model.eval()
            # Forward pass
            after_micro_batch = input_sample[[i]].to(device).requires_grad_(True)
            output = method.model(after_micro_batch)
            probs = F.softmax(output, dim=1)

            # Compute entropy
            entropy = compute_entropy(probs)
            after_entropy_values.append(entropy.item())

            # Compute cross-entropy loss
            cross_entropy_loss = compute_cross_entropy(output, label[[i]].to(device))
            after_cross_entropy_values.append(cross_entropy_loss.item())

            # Use entropy (or max logit) as pseudo-loss to compute gradients
            pseudo_loss = entropy.sum()
            pseudo_loss.backward(retain_graph=True)

            # Compute gradient norm w.r.t. input
            grad_norm = compute_gradient_norm(entropy, after_micro_batch)
            after_gradient_norms.append(grad_norm.item())
            # print("after", grad_norm.item())

            method.model.zero_grad()
    return (
        before_entropy_values,
        before_gradient_norms,
        before_cross_entropy_values,
        after_entropy_values,
        after_gradient_norms,
        after_cross_entropy_values,
    )


(
    before_entropy_values,
    before_gradient_norms,
    before_cross_entropy_values,
    after_entropy_values,
    after_gradient_norms,
    after_cross_entropy_values,
) = calculate()

st.title("Interactive Comparison of Entropy, Gradient Norm, and Cross Entropy")


# Function to create the interactive plot
def create_plot(
    column,
    gradient_norms,
    entropy_values,
    cross_entropy_values,
    title,
):
    with column:
        st.markdown(f"## {title}")
        # Dropdowns for selecting x and y axes
        x_axis = st.selectbox(
            "Select X-axis",
            ["Gradient Norm", "Entropy", "Cross Entropy"],
            key=f"x_axis_{column}",
        )
        y_axis = st.selectbox(
            "Select Y-axis",
            ["Entropy", "Gradient Norm", "Cross Entropy"],
            key=f"y_axis_{column}",
        )

        # Mapping selections to data
        data_map = {
            "Gradient Norm": gradient_norms,
            "Entropy": entropy_values,
            "Cross Entropy": cross_entropy_values,
        }

        fig, ax = plt.subplots(figsize=(9, 6))

        x_data = data_map[x_axis]
        y_data = data_map[y_axis]
        # label = labels

        ax.scatter(
            x_data,
            y_data,
            # color=colors,
            # label=label,
            alpha=0.7,
        )
        for i in range(len(x_data)):
            ax.annotate(
                f"{i}",
                (x_data[i], y_data[i]),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
            )
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{y_axis} vs. {x_axis}")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


# Create two columns
col1, col2 = st.columns(2)

# Create plots in both columns
create_plot(
    column=col1,
    gradient_norms=before_gradient_norms,
    entropy_values=before_entropy_values,
    cross_entropy_values=before_cross_entropy_values,
    title="before adaptation",
)
create_plot(
    column=col2,
    gradient_norms=after_gradient_norms,
    entropy_values=after_entropy_values,
    cross_entropy_values=after_cross_entropy_values,
    title="after adaptation",
)
if st.sidebar.button("Run Adaptation"):
    calculate.clear()
    st.rerun()
