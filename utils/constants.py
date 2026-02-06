# Shared constants

# HU/CT clipping range
CLIP_MIN = -1024
CLIP_MAX = 3000

# Stats files
STATS_DIRNAME = "stats"
MEAN_FILENAME = "mean.npy"
STD_FILENAME = "std.npy"
METADATA_FILENAME = "stats_meta.json"

# Common filenames
MASK_FILENAME = "mask.mha"
CT_FILENAME = "ct.mha"
CT_PREPROC_FILENAME = "ct_preproc.mha"
CBCT_FILENAME = "cbct.mha"
CBCT_PREPROC_FILENAME = "cbct_preproc.mha"
MR_FILENAME = "mr.mha"
MR_PREPROC_FILENAME = "mr_preproc.mha"
SCT_FILENAME = "sct.mha"
SCT_POSTPROC_FILENAME = "sct_postproc.mha"

# 3D patch sizes by anatomy
PATCH_SIZE_3D_BY_ANATOMY = {
    "AB": (32, 192, 224),
    "HN": (32, 192, 160),
    "TH": (32, 192, 192),
}
