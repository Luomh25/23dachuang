import os
import glob
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import filters


# 定义输入和输出文件夹路径
input_folder = 'C:/Users/Lucille/Desktop/dachuang/biostat/ADNI_MRI/test\ADNI1_Complete 3Yr 1.5T/ADNI/'
output_folder = 'C:/Users/Lucille/Desktop/dachuang/biostat/ADNI_MRI/test\ADNI1_Complete 3Yr 1.5T/ADNI/test/'

# 获取输入文件夹中的所有nii文件路径
input_files = glob.glob(os.path.join(input_folder, '*.nii'))

# 循环处理每个输入文件
for input_file in input_files:
    # 加载MRI数据
    image = nib.load(input_file)
    data = image.get_fdata()

    # 预处理步骤（配准和脑组织分割）
    processed_data = ndimage.median_filter(data, size=3)
    threshold = filters.threshold_otsu(processed_data)#阈值分割
    brain_mask = processed_data > threshold
    brain_mask = ndimage.binary_fill_holes(brain_mask)

    # 构建输出文件路径
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_folder, filename)

    # 保存处理后的数据和脑掩膜
    processed_image = nib.Nifti1Image(processed_data, image.affine)
    nib.save(processed_image, output_file)

    mask_image = nib.Nifti1Image(brain_mask.astype(np.uint8), image.affine)
    mask_output_file = os.path.join(output_folder, 'mask_' + filename)
    nib.save(mask_image, mask_output_file)