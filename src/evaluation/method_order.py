"""Shared left-to-right method order for evaluation and saved-result plots."""


def ordered_methods(names):
    """Order selected methods as TFI, PDE, MeshONet, HMONet.

    Keep variants within each family in their original relative order, and
    retain unrecognized methods at the end without adding missing methods.
    """

    def rank(name):
        key = str(name).strip().lower()
        for index, family in enumerate(
            ("tfi", "pde", "meshonet", "pi-meshonet", "hmonet", "pi-hmonet", "attention-pi-hmonet")
        ):
            if key.startswith(family):
                return index
        return 7

    return sorted(names, key=rank)


def method_label(name):
    """Use the short PDE plot title while preserving the saved method key."""
    return "PDE" if name == "PDE_Winslow" else name
