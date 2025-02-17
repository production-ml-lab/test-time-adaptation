from torch.utils.data import DataLoader


def build_loader(dataset, batch_size: int = 16, sampler: str = "base"):
    # TODO: Add Dirichlet sampler
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader
