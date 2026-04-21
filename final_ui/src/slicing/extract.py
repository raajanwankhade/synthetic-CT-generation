def extract_axial_slices(case, indices):
    """
    Extract axial slices for all modalities using same indices.

    Args:
        case: dict from load_case()
        indices: list of slice indices

    Returns:
        dict with lists of 2D slices
    """
    slices = {
        "ct": [],
        "sct": [],
        "mri": [],
        "gradcam": []
    }

    for i in indices:
        slices["ct"].append(case["ct"][i])
        slices["sct"].append(case["sct"][i])
        slices["mri"].append(case["mri"][i])
        slices["gradcam"].append(case["gradcam"][i])

    return slices
