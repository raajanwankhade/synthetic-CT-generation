import SimpleITK as sitk


def load_image_as_array(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def write_array_as_image(arr, ref_img, out_path, use_compression=True):
    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, out_path, useCompression=use_compression)


def load_medical_image(path):
    return load_image_as_array(path)
