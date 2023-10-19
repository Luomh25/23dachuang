#!/usr/bin/env python

# Date: 08/02/2021
# Author: Christoph Arthofer
# Copyright: FMRIB 2021

import os
import shutil
import pandas as pd
import nibabel as nib
import numpy as np
import shlex
import subprocess
#from fsl.wrappers import fslmaths,flirt,concatxfm,fslstats,bet,fast,invxfm,invwarp
#from fsl.wrappers.fnirt import applywarp, convertwarp
#from file_tree import FileTree
#from fsl.utils.fslsub import func_to_cmd
#import fsl_sub
#import fsl_sub.utils
from operator import itemgetter
import tempfile
import argparse
import sys

def writeConfig2mm(mod,fpath):
    T1_brain = True if all(m in mod for m in ['T1_brain']) else False
    T1_head = True if any(m in mod for m in ['T1_head']) else False
    T2_head = True if any(m in mod for m in ['T2_head']) else False
    DTI = True if all(m in mod for m in ['DTI']) else False

    s = 'warp_res_init           = 32 \n' \
        'warp_scaling            = 1 1 2 2 2 2 \n' \
        'lambda_reg              = 4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1 1.8e-1 \n' \
        'hires                   = 3.9 \n' \
        'optimiser_max_it_lowres = 5 \n' \
        'optimiser_max_it_hires  = 5 \n'
    if T1_head:
        s += '\n' \
        '; Whole-head T1 \n' \
        'use_implicit_mask       = 0 \n' \
        'use_mask_ref_scalar     = 1 1 1 1 1 1 \n' \
        'use_mask_mov_scalar     = 0 0 0 0 0 0 \n' \
        'fwhm_ref_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'fwhm_mov_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'lambda_scalar           = 1 1 1 1 1 1 \n' \
        'estimate_bias           = 1 \n' \
        'bias_res_init           = 32 \n' \
        'lambda_bias_reg         = 1e9 1e9 1e9 1e9 1e9 1e9 \n'
    if T1_brain:
        s += '\n' \
        '; Brain-only T1 \n' \
        'use_implicit_mask       = 0 \n' \
        'use_mask_ref_scalar     = 0 0 0 0 0 0 \n' \
        'use_mask_mov_scalar     = 0 0 0 0 0 0 \n' \
        'fwhm_ref_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'fwhm_mov_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'lambda_scalar           = 1 1 1 1 1 1 \n' \
        'estimate_bias           = 0 \n' \
        'bias_res_init           = 32 \n' \
        'lambda_bias_reg         = 1e9 1e9 1e9 1e9 1e9 1e9 \n'
    if T2_head:
        s += '\n' \
        '; Whole-head T2 \n' \
        'use_implicit_mask       = 0 \n' \
        'use_mask_ref_scalar     = 0 0 0 0 0 0 \n' \
        'use_mask_mov_scalar     = 0 0 0 0 0 0 \n' \
        'fwhm_ref_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'fwhm_mov_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'lambda_scalar           = 1 1 1 1 1 1 \n' \
        'estimate_bias           = 1 \n' \
        'bias_res_init           = 32 \n' \
        'lambda_bias_reg         = 1e9 1e9 1e9 1e9 1e9 1e9 \n'
    if DTI:
        s += '\n' \
        '; DTI \n' \
        'use_mask_ref_tensor     = 1 1 1 1 1 1 \n' \
        'use_mask_mov_tensor     = 0 0 0 0 0 0 \n' \
        'fwhm_ref_tensor         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'fwhm_mov_tensor         = 8.0 8.0 4.0 2.0 1.0 0.5 \n' \
        'lambda_tensor           = 1 1 1 1 1 1 \n'

    f = open(fpath, 'w')
    f.write(s)
    f.close()

def writeConfig1mm(mod,fpath):
    T1_brain = True if all(m in mod for m in ['T1_brain']) else False
    T1_head = True if any(m in mod for m in ['T1_head']) else False
    T2_head = True if any(m in mod for m in ['T2_head']) else False
    DTI = True if all(m in mod for m in ['DTI']) else False

    s = 'warp_res_init           = 32 \n' \
        'warp_scaling            = 1 1 2 2 2 2 2 \n' \
        'lambda_reg              = 4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1 1.8e-1 1.5e-1 \n' \
        'hires                   = 3.9 \n' \
        'optimiser_max_it_lowres = 5 \n' \
        'optimiser_max_it_hires  = 5 \n'
    if T1_head:
        s += '\n' \
        '; Whole-head T1 \n' \
        'use_implicit_mask       = 0 \n' \
        'use_mask_ref_scalar     = 1 1 1 1 1 1 1 \n' \
        'use_mask_mov_scalar     = 0 0 0 0 0 0 0 \n' \
        'fwhm_ref_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'fwhm_mov_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'lambda_scalar           = 1 1 1 1 1 1 1 \n' \
        'estimate_bias           = 1 \n' \
        'bias_res_init           = 32 \n' \
        'lambda_bias_reg         = 1e9 1e9 1e9 1e9 1e9 1e9 1e9 \n'
    if T1_brain:
        s += '\n' \
        '; Brain-only T1 \n' \
        'use_implicit_mask       = 0 \n' \
        'use_mask_ref_scalar     = 0 0 0 0 0 0 0 \n' \
        'use_mask_mov_scalar     = 0 0 0 0 0 0 0 \n' \
        'fwhm_ref_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'fwhm_mov_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'lambda_scalar           = 1 1 1 1 1 1 1 \n' \
        'estimate_bias           = 0 \n' \
        'bias_res_init           = 32 \n' \
        'lambda_bias_reg         = 1e9 1e9 1e9 1e9 1e9 1e9 1e9 \n'
    if T2_head:
        s += '\n' \
        '; Whole-head T2 \n' \
        'use_implicit_mask       = 0 \n' \
        'use_mask_ref_scalar     = 0 0 0 0 0 0 0 \n' \
        'use_mask_mov_scalar     = 0 0 0 0 0 0 0 \n' \
        'fwhm_ref_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'fwhm_mov_scalar         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'lambda_scalar           = 1 1 1 1 1 1 1 \n' \
        'estimate_bias           = 1 \n' \
        'bias_res_init           = 32 \n' \
        'lambda_bias_reg         = 1e9 1e9 1e9 1e9 1e9 1e9 1e9 \n'
    if DTI:
        s += '\n' \
        '; DTI \n' \
        'use_mask_ref_tensor     = 1 1 1 1 1 1 1 \n' \
        'use_mask_mov_tensor     = 0 0 0 0 0 0 0 \n' \
        'fwhm_ref_tensor         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'fwhm_mov_tensor         = 8.0 8.0 4.0 2.0 1.0 0.5 0.25 \n' \
        'lambda_tensor           = 1 1 1 1 1 1 1 \n'

    f = open(fpath, 'w')
    f.write(s)
    f.close()

def soft_clamp(x,k):
    # Piecewise function for soft intensity clamping of T1 images.
    # Takes a single parameter k which defines the transition to the clamping part of the
    # function
    #
    # f(x) = 0                                  | x <= 0
    # f(x) = x                                  | 0 < x <= k
    # f(x) = 3k/4 + k/(2(1 + exp(-8(x - k)/k))) | x > k
    # Date: 08/02/2021
    # Author: Frederik J Lange
    # Copyright: FMRIB 2021

    return np.piecewise(x,
                        [x <= 0, (0 < x) & (x <= k), x > k],
                        [lambda x: 0, lambda x: x, lambda x: k/(2*(1 + np.exp(-8*(x - k)/k)))+ 0.75*k])

def clampImage(img_path,out_path):
    out_dir = os.path.split(out_path)[0]
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdirname:
        mask_path = os.path.splitext(os.path.splitext(os.path.basename(out_path))[0])[0] + '_brain.nii.gz'
        mask_path = os.path.join(out_dir, mask_path)
        bet(img_path, mask_path, robust=True)
        fast(mask_path, tmpdirname + '/fast', iter=0, N=True, g=True, v=False)
        wm_intensity_mean = fslstats(mask_path).k(tmpdirname + '/fast_seg_2').M.run()
        print('White matter mean intensity is: ', wm_intensity_mean)

    img_nib = nib.load(img_path)
    img_clamped_np = soft_clamp(img_nib.get_fdata(),wm_intensity_mean)
    img_clamped_nib = nib.Nifti1Image(img_clamped_np, affine=img_nib.affine, header=img_nib.header)
    img_clamped_nib.to_filename(out_path)

def applyWarpWrapper(src, ref, out, warp, premat=None, interp='spline', norm_bool=False):
    if os.path.exists(src) and os.path.exists(ref) and os.path.exists(warp):
        img_nib = nib.load(src)
        if norm_bool:
            img_nib = fslmaths(img_nib).inm(1000).run()
        print('applywarp(src=img_nib,ref=ref_path,out=warped_path,warp=warp_path,interp=interp)')
        if premat is None:
            applywarp(src=img_nib,ref=ref,out=out,warp=warp,interp=interp)
        elif premat is not None:
            applywarp(src=img_nib, ref=ref, out=out, warp=warp, premat=premat, interp=interp)
        return 1
    else:
        print('One or more of: '+src+'\n'+ref+'\n'+warp+' do not exist!')
        return 0


def fslsubWrapper(command, name, log_dir, queue, wait_for=None, array_task=False, coprocessor=None, coprocessor_class=None, coprocessor_multi="1", threads=1, export_var=None):
    coprocessor_class_strict = True if coprocessor_class is not None else False

    job_id = fsl_sub.submit(command=command,
                   array_task=array_task,
                   jobhold=wait_for,
                   name=name,
                   logdir=log_dir,
                   queue=queue,
                   coprocessor=coprocessor,
                   coprocessor_class=coprocessor_class,
                   coprocessor_class_strict=coprocessor_class_strict,
                   coprocessor_multi=coprocessor_multi,
                   threads=threads,
                   export_vars=export_var
                   )

    return job_id

def calcVolume(img_path, threshold):
    if os.path.exists(img_path):
        # mask_nib = fslmaths(img_path).thr(threshold).run()
        # voxels, vol = fslstats(mask_nib).V.run()

        mask_nib = nib.load(img_path)
        mask_hdr = mask_nib.header
        pixdim = mask_hdr['pixdim']

        mask_img = mask_nib.get_fdata().flatten()
        n_voxels = np.sum(mask_img[mask_img > threshold])
        vol = n_voxels * np.prod(pixdim[1:4])

        return vol
    else:
        return np.nan

def calcIntensity(img_path, label_path, threshold):
    if os.path.exists(img_path) and os.path.exists(label_path):

        img_np = nib.load(img_path).get_fdata()
        label_np = nib.load(label_path).get_fdata()

        label_np[label_np < threshold] = 0

        img_np_masked = img_np[label_np > 0].flatten()
        ob_sum = np.sum(img_np_masked)
        ob_mean = np.mean(img_np_masked)
        ob_median = np.median(img_np_masked)
        ob_max = np.amax(img_np_masked)
        ob_percentile_90 = np.percentile(img_np_masked,90)
        ob_percentile_95 = np.percentile(img_np_masked,95)

        return {'mean': ob_mean,
                'median': ob_median,
                'max': ob_max,
                '90th_percentile': ob_percentile_90,
                '95th_percentile': ob_percentile_95,
                'sum': ob_sum
                }
    else:
        return {'mean': np.nan,
                'median': np.nan,
                'max': np.nan,
                '90th_percentile': np.nan,
                '95th_percentile': np.nan,
                'sum': np.nan
                }

def concatVertically(csv_paths, output_path):
    for i, path in enumerate(csv_paths):
        if os.path.exists(path):
            if i == 0:
                df_temp = pd.read_csv(path)
            else:
                df_temp = pd.concat([df_temp, pd.read_csv(path)])

    df_temp.to_csv(output_path,header=True,index=False,na_rep='nan')

def mmorfWrapper(mmorf_run_cmd,config_path,img_warp_space,img_ref_scalar,img_mov_scalar,aff_ref_scalar,aff_mov_scalar,mask_ref_scalar,
                     mask_mov_scalar,img_ref_tensor,img_mov_tensor,aff_ref_tensor,aff_mov_tensor,mask_ref_tensor,mask_mov_tensor,
                     warp_out,jac_det_out,bias_out):
    export_var = []
    cmd = mmorf_run_cmd
    cmd += ' --config ' + config_path
    split = os.path.split(config_path)
    export_var.append(split[0])
    cmd += ' --img_warp_space ' + img_warp_space
    split = os.path.split(img_warp_space)
    export_var.append(split[0])
    for path in img_ref_scalar:
        cmd += ' --img_ref_scalar ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in img_mov_scalar:
        cmd += ' --img_mov_scalar ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in aff_ref_scalar:
        cmd += ' --aff_ref_scalar ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in aff_mov_scalar:
        cmd += ' --aff_mov_scalar ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in mask_ref_scalar:
        cmd += ' --mask_ref_scalar ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in mask_mov_scalar:
        cmd += ' --mask_mov_scalar ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in img_ref_tensor:
        cmd += ' --img_ref_tensor ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in img_mov_tensor:
        cmd += ' --img_mov_tensor ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in aff_ref_tensor:
        cmd += ' --aff_ref_tensor ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in aff_mov_tensor:
        cmd += ' --aff_mov_tensor ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in mask_ref_tensor:
        cmd += ' --mask_ref_tensor ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    for path in mask_mov_tensor:
        cmd += ' --mask_mov_tensor ' + path
        split = os.path.split(path)
        export_var.append(split[0])
    cmd += ' --warp_out ' + warp_out
    split = os.path.split(warp_out)
    export_var.append(split[0])
    cmd += ' --jac_det_out ' + jac_det_out
    split = os.path.split(jac_det_out)
    export_var.append(split[0])
    cmd += ' --bias_out ' + bias_out
    split = os.path.split(bias_out)
    export_var.append(split[0])

    cmd += '\n'

    export_var = list(filter(None,list(set(export_var))))
    export_var = {'SINGULARITY_BIND':export_var}

    return cmd, export_var

def affinePart(tree, clamping_on=False):
    if clamping_on:
        T1_head_path = tree.get('data/T1_head')
        T1_clamped_path = tree.get('T1_head_clamped', make_dir=True)
        clampImage(T1_head_path, T1_clamped_path)

    flirt(tree.get('data/T1_brain'), tree.get('T1_brain_nln_template'), omat=tree.get('T1_brain_to_template_mat', make_dir=True), dof=12)
    flirt(tree.get('data/T2_brain'), tree.get('data/T1_brain'), omat=tree.get('T2_to_T1_mat'), dof=6)
    flirt(tree.get('data/DTI_scalar'), tree.get('data/T2_brain'), omat=tree.get('DTI_to_T2_mat'), dof=6)
    concatxfm(tree.get('T2_to_T1_mat'), tree.get('T1_brain_to_template_mat'), tree.get('T2_brain_to_template_mat'))
    concatxfm(tree.get('DTI_to_T2_mat'), tree.get('T2_brain_to_template_mat'), tree.get('DTI_to_template_mat'))
    invxfm(tree.get('T1_brain_to_template_mat'), tree.get('T1_brain_to_template_invmat'))
    invxfm(tree.get('T2_brain_to_template_mat'), tree.get('T2_brain_to_template_invmat'))
    invxfm(tree.get('T2_to_T1_mat'), tree.get('T2_to_T1_invmat'))
    invxfm(tree.get('DTI_to_template_mat'), tree.get('DTI_to_template_invmat'))
    concatxfm(tree.get('T1_brain_to_template_invmat'), tree.get('data/T1_to_SWI_mat'), tree.get('T2_star_to_template_invmat'))

    return 1

def createInverseWarps(tree):
# Inverse warp
    if os.path.exists(tree.get('mmorf_warp')) and os.path.exists(tree.get('T1_head_nln_template')):
        invwarp(warp=tree.get('mmorf_warp'), ref=tree.get('T1_head_nln_template'), out=tree.get('mmorf_invwarp'))
    else:
        return 0
# Concatenate corresponding affine transforms and warps
    if os.path.exists(tree.get('data/T2_head')) and os.path.exists(tree.get('T2_brain_to_template_invmat')) and os.path.exists(tree.get('mmorf_invwarp')):
        convertwarp(out=tree.get('mmorf_invwarp_comp_T2'), ref=tree.get('data/T2_head'), postmat=tree.get('T2_brain_to_template_invmat'), warp1=tree.get('mmorf_invwarp'))

    if os.path.exists(tree.get('data/T2_star')) and os.path.exists(tree.get('T2_star_to_template_invmat')) and os.path.exists(tree.get('mmorf_invwarp')):
        convertwarp(out=tree.get('mmorf_invwarp_comp_T2_star'), ref=tree.get('data/T2_star'), postmat=tree.get('T2_star_to_template_invmat'), warp1=tree.get('mmorf_invwarp'))

    if os.path.exists(tree.get('data/DTI_scalar')) and os.path.exists(tree.get('DTI_to_template_invmat')) and os.path.exists(tree.get('mmorf_invwarp')):
        convertwarp(out=tree.get('mmorf_invwarp_comp_DTI'), ref=tree.get('data/DTI_scalar'), postmat=tree.get('DTI_to_template_invmat'), warp1=tree.get('mmorf_invwarp'))

    if os.path.exists(tree.get('data/T2_star')) and os.path.exists(tree.get('T2_star_to_template_invmat')) and os.path.exists(tree.get('OXMM_to_MNI_invwarp_comp')) and os.path.exists(tree.get('mmorf_invwarp')):
        convertwarp(out=tree.get('mmorf_MNI_invwarp_comp_T2_star'), ref=tree.get('data/T2_star'), postmat=tree.get('T2_star_to_template_invmat'), warp1=tree.get('OXMM_to_MNI_invwarp_comp'), warp2=tree.get('mmorf_invwarp'))
    else:
        print('Error in T2star warp composition!')
    if os.path.exists(tree.get('data/DTI_scalar')) and os.path.exists(tree.get('DTI_to_template_invmat')) and os.path.exists(tree.get('OXMM_to_MNI_invwarp_comp')) and os.path.exists(tree.get('mmorf_invwarp')):
        convertwarp(out=tree.get('mmorf_MNI_invwarp_comp_DTI'), ref=tree.get('data/DTI_scalar'), postmat=tree.get('DTI_to_template_invmat'), warp1=tree.get('OXMM_to_MNI_invwarp_comp'), warp2=tree.get('mmorf_invwarp'))
    else:
        print('Error in DTI warp composition!')

    convertwarp(out=tree.get('mmorf_MNI_invwarp_comp_T1'), ref=tree.get('data/T1_head'), postmat=tree.get('T1_brain_to_template_invmat'), warp1=tree.get('OXMM_to_MNI_invwarp_comp'), warp2=tree.get('mmorf_invwarp'))

    return 1

def warpMasksToOXMM(tree):
# AON to DTI
    applyWarpWrapper(src=tree.get('AON_T1_MNI_label'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('AON_T1_OXMM_label'),
                     warp=tree.get('OXMM_to_MNI_invwarp_comp'), interp='trilinear', norm_bool=False)
# PIF to DTI
    applyWarpWrapper(src=tree.get('PIF_T1_MNI_label'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('PIF_T1_OXMM_label'),
                     warp=tree.get('OXMM_to_MNI_invwarp_comp'), interp='trilinear', norm_bool=False)
# PIT to DTI
    applyWarpWrapper(src=tree.get('PIT_T1_MNI_label'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('PIT_T1_OXMM_label'),
                     warp=tree.get('OXMM_to_MNI_invwarp_comp'), interp='trilinear', norm_bool=False)
# TUB to DTI
    applyWarpWrapper(src=tree.get('TUB_T1_MNI_label'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('TUB_T1_OXMM_label'),
                     warp=tree.get('OXMM_to_MNI_invwarp_comp'), interp='trilinear', norm_bool=False)

    return 1

def warpMasksToSubjects(tree):
# AON to T1
    applyWarpWrapper(src=tree.get('AON_T1_MNI_label'), ref=tree.get('data/T1_head'),
                     out=tree.get('warped_AON_label_in_T1_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T1'), interp='trilinear', norm_bool=False)
# PIF to T1
    applyWarpWrapper(src=tree.get('PIF_T1_MNI_label'), ref=tree.get('data/T1_head'),
                     out=tree.get('warped_PIF_label_in_T1_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T1'), interp='trilinear', norm_bool=False)
# PIT to T1
    applyWarpWrapper(src=tree.get('PIT_T1_MNI_label'), ref=tree.get('data/T1_head'),
                     out=tree.get('warped_PIT_label_in_T1_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T1'), interp='trilinear', norm_bool=False)
# TUB to T1
    applyWarpWrapper(src=tree.get('TUB_T1_MNI_label'), ref=tree.get('data/T1_head'),
                     out=tree.get('warped_TUB_label_in_T1_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T1'), interp='trilinear', norm_bool=False)

# Apply warps to masks
# left OB to T2
    applyWarpWrapper(src=tree.get('OB_left_label'), ref=tree.get('data/T2_head'),
                     out=tree.get('warped_OB_left_label_in_T2_native'),
                     warp=tree.get('mmorf_invwarp_comp_T2'), interp='trilinear', norm_bool=False)
# right OB to T2
    applyWarpWrapper(src=tree.get('OB_right_label'), ref=tree.get('data/T2_head'),
                     out=tree.get('warped_OB_right_label_in_T2_native'),
                     warp=tree.get('mmorf_invwarp_comp_T2'), interp='trilinear', norm_bool=False)
# PG to T2
    applyWarpWrapper(src=tree.get('PG_label'), ref=tree.get('data/T2_head'),
                     out=tree.get('warped_PG_label_in_T2_native'),
                     warp=tree.get('mmorf_invwarp_comp_T2'), interp='trilinear', norm_bool=False)
# HT to T2
    applyWarpWrapper(src=tree.get('HT_label'), ref=tree.get('data/T2_head'),
                     out=tree.get('warped_HT_label_in_T2_native'),
                     warp=tree.get('mmorf_invwarp_comp_T2'), interp='trilinear', norm_bool=False)

# PG to T2*
    applyWarpWrapper(src=tree.get('PG_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_PG_label_in_T2_star_native'),
                     warp=tree.get('mmorf_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)
# HT to T2*
    applyWarpWrapper(src=tree.get('HT_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_HT_label_in_T2_star_native'),
                     warp=tree.get('mmorf_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)
# SN to T2*
    applyWarpWrapper(src=tree.get('SN_MNI_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_SN_label_in_T2_star_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)
# OC (olfactory cortex) to T2*
    applyWarpWrapper(src=tree.get('OC_MNI_1mm_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_OC_label_in_T2_star_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)
# AON_T1 to T2*
    applyWarpWrapper(src=tree.get('AON_T1_MNI_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_AON_label_in_T2_star_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)
# PIF_T1 to T2*
    applyWarpWrapper(src=tree.get('PIF_T1_MNI_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_PIF_label_in_T2_star_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)
# PIT_T1 to T2*
    applyWarpWrapper(src=tree.get('PIT_T1_MNI_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_PIT_label_in_T2_star_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)
# TUB_T1 to T2*
    applyWarpWrapper(src=tree.get('TUB_T1_MNI_label'), ref=tree.get('data/T2_star'),
                     out=tree.get('warped_TUB_label_in_T2_star_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_T2_star'), interp='trilinear', norm_bool=False)


# PG to DTI
    applyWarpWrapper(src=tree.get('PG_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_PG_label_in_DTI_native'),
                     warp=tree.get('mmorf_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)
# HT to DTI
    applyWarpWrapper(src=tree.get('HT_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_HT_label_in_DTI_native'),
                     warp=tree.get('mmorf_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)
# SN to DTI
    applyWarpWrapper(src=tree.get('SN_MNI_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_SN_label_in_DTI_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)
# OC (olfactory cortex) to DTI
    applyWarpWrapper(src=tree.get('OC_MNI_1mm_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_OC_label_in_DTI_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)
# AON to DTI
    applyWarpWrapper(src=tree.get('AON_T1_MNI_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_AON_label_in_DTI_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)
# PIF to DTI
    applyWarpWrapper(src=tree.get('PIF_T1_MNI_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_PIF_label_in_DTI_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)
# PIT to DTI
    applyWarpWrapper(src=tree.get('PIT_T1_MNI_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_PIT_label_in_DTI_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)
# TUB to DTI
    applyWarpWrapper(src=tree.get('TUB_T1_MNI_label'), ref=tree.get('data/DTI_scalar'),
                     out=tree.get('warped_TUB_label_in_DTI_native'),
                     warp=tree.get('mmorf_MNI_invwarp_comp_DTI'), interp='trilinear', norm_bool=False)


# Warps to template
# FAST GM mask to template
    applyWarpWrapper(src=tree.get('data/T1_brain_pve1_mask'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('warped_T1_brain_pve1_mask'),
                     warp=tree.get('mmorf_warp'), premat=tree.get('T1_brain_to_template_mat'), interp='trilinear', norm_bool=False)

# DTI_MD, DTI_FA, NODDI output to template
    applyWarpWrapper(src=tree.get('data/DTI_FA'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('warped_DTI_FA'),
                     warp=tree.get('mmorf_warp'), premat=tree.get('DTI_to_template_mat'), interp='spline', norm_bool=False)

    applyWarpWrapper(src=tree.get('data/DTI_MD'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('warped_DTI_MD'),
                     warp=tree.get('mmorf_warp'), premat=tree.get('DTI_to_template_mat'), interp='spline', norm_bool=False)

    applyWarpWrapper(src=tree.get('data/NODDI_ICVF'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('warped_NODDI_ICVF'),
                     warp=tree.get('mmorf_warp'), premat=tree.get('DTI_to_template_mat'), interp='spline', norm_bool=False)

    applyWarpWrapper(src=tree.get('data/NODDI_ISOVF'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('warped_NODDI_ISOVF'),
                     warp=tree.get('mmorf_warp'), premat=tree.get('DTI_to_template_mat'), interp='spline', norm_bool=False)

    applyWarpWrapper(src=tree.get('data/NODDI_OD'), ref=tree.get('T1_head_nln_template'),
                     out=tree.get('warped_NODDI_OD'),
                     warp=tree.get('mmorf_warp'), premat=tree.get('DTI_to_template_mat'), interp='spline', norm_bool=False)

    return 1

def runRegression(in_path, d_path, o_path):
    if os.path.exists(in_path) and os.path.exists(d_path):
        cmd = 'fsl_glm -i ' + in_path + ' -d ' + d_path + ' -o ' + o_path
        try:
            subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
            arr = np.loadtxt(o_path)
            return arr.flatten()[0]
        except subprocess.CalledProcessError as e:
            print(str(e), file=sys.stderr)
            return np.nan
    else:
        return np.nan

def extractIDPs(tree, subjectid, warp_resolution):
    tree = tree.update(sub_id=subjectid)
# Take measurements
    idps = {'subjectID': subjectid}

    thresholds = [0,0.3];
    for threshold in thresholds:
# Calculate volume
        idps['OB_left_volume_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = calcVolume(img_path=tree.get('warped_OB_left_label_in_T2_native'), threshold=threshold)
        idps['OB_right_volume_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = calcVolume(img_path=tree.get('warped_OB_right_label_in_T2_native'), threshold=threshold)
        idps['HT_volume_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = calcVolume(img_path=tree.get('warped_HT_label_in_T2_native'), threshold=threshold)
        idps['PG_volume_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = calcVolume(img_path=tree.get('warped_PG_label_in_T2_native'), threshold=threshold)

# Calculate intensity IDPs in T2
        anat_structs = ['OB_left', 'OB_right', 'PG', 'HT']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/T2_head'), label_path=tree.get('warped_'+s+'_label_in_T2_native'), threshold=threshold)
            idps[s+'_mean_intensity_mod=T2_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['mean']
            idps[s+'_median_intensity_mod=T2_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['median']
            idps[s+'_max_intensity_mod=T2_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['max']
            idps[s+'_90th_percentile_mod=T2_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['90th_percentile']
            idps[s+'_95th_percentile_mod=T2_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['95th_percentile']

# Calculate intensity IDPs in T2*
        anat_structs = ['PG', 'HT', 'SN', 'OC']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/T2_star'), label_path=tree.get('warped_'+s+'_label_in_T2_star_native'), threshold=threshold)
            idps[s+'_mean_intensity_mod=T2_star_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['mean']
            idps[s+'_median_intensity_mod=T2_star_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['median']
            idps[s+'_max_intensity_mod=T2_star_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['max']
            idps[s+'_90th_percentile_mod=T2_star_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['90th_percentile']
            idps[s+'_95th_percentile_mod=T2_star_threshold='+str(threshold)+'_warpResolution='+warp_resolution] = intensity_dict['95th_percentile']

# Calculate intensity IDPs in DTI FA
        anat_structs = ['PG', 'HT', 'SN', 'OC']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/DTI_FA'), label_path=tree.get('warped_'+s+'_label_in_DTI_native'), threshold=threshold)
            idps[s+'_mean_intensity_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s+'_median_intensity_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s+'_max_intensity_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s+'_90th_percentile_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s+'_95th_percentile_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']

# Calculate intensity IDPs in DTI MD
        anat_structs = ['PG', 'HT', 'SN', 'OC']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/DTI_MD'), label_path=tree.get('warped_'+s+'_label_in_DTI_native'), threshold=threshold)
            idps[s+'_mean_intensity_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s+'_median_intensity_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s+'_max_intensity_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s+'_90th_percentile_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s+'_95th_percentile_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']

# Calculate intensity IDPs in NODDI ICVF
        anat_structs = ['PG', 'HT', 'SN', 'OC']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/NODDI_ICVF'), label_path=tree.get('warped_'+s+'_label_in_DTI_native'), threshold=threshold)
            idps[s+'_mean_intensity_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s+'_median_intensity_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s+'_max_intensity_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s+'_90th_percentile_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s+'_95th_percentile_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']

# Calculate intensity IDPs in NODDI ISOVF
        anat_structs = ['PG', 'HT', 'SN', 'OC']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/NODDI_ISOVF'), label_path=tree.get('warped_'+s+'_label_in_DTI_native'), threshold=threshold)
            idps[s+'_mean_intensity_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s+'_median_intensity_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s+'_max_intensity_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s+'_90th_percentile_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s+'_95th_percentile_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']

# Calculate intensity IDPs in NODDI OD
        anat_structs = ['PG', 'HT', 'SN', 'OC']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/NODDI_OD'), label_path=tree.get('warped_'+s+'_label_in_DTI_native'), threshold=threshold)
            idps[s+'_mean_intensity_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s+'_median_intensity_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s+'_max_intensity_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s+'_90th_percentile_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s+'_95th_percentile_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']

    thresholds = [0]
    for threshold in thresholds:
# Calculate intensity IDPs in T2*
        anat_structs = ['AON', 'PIF', 'PIT', 'TUB']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/T2_star'), label_path=tree.get('warped_' + s + '_label_in_T2_star_native'), threshold=threshold)
            idps[s + '_mean_intensity_mod=T2_star_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s + '_median_intensity_mod=T2_star_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s + '_max_intensity_mod=T2_star_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s + '_90th_percentile_mod=T2_star_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s + '_95th_percentile_mod=T2_star_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']
            idps[s + '_glm_mod=T2_star_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = \
                runRegression(in_path=tree.get('data/T2_star'), d_path=tree.get('warped_' + s + '_label_in_T2_star_native'), o_path=tree.get('temp_GLM_output'))

# Calculate intensity IDPs in DTI FA
        anat_structs = ['AON', 'PIF', 'PIT', 'TUB']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/DTI_FA'), label_path=tree.get('warped_' + s + '_label_in_DTI_native'), threshold=threshold)
            idps[s + '_mean_intensity_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s + '_median_intensity_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s + '_max_intensity_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s + '_90th_percentile_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s + '_95th_percentile_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']
            idps[s + '_glm_mod=DTI_FA_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = \
                runRegression(in_path=tree.get('data/DTI_FA'), d_path=tree.get('warped_' + s + '_label_in_DTI_native'), o_path=tree.get('temp_GLM_output'))

# Calculate intensity IDPs in DTI MD
        anat_structs = ['AON', 'PIF', 'PIT', 'TUB']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/DTI_MD'), label_path=tree.get('warped_' + s + '_label_in_DTI_native'), threshold=threshold)
            idps[s + '_mean_intensity_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s + '_median_intensity_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s + '_max_intensity_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s + '_90th_percentile_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s + '_95th_percentile_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']
            idps[s + '_glm_mod=DTI_MD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = \
                runRegression(in_path=tree.get('data/DTI_MD'), d_path=tree.get('warped_' + s + '_label_in_DTI_native'), o_path=tree.get('temp_GLM_output'))

# Calculate intensity IDPs in NODDI ICVF
        anat_structs = ['AON', 'PIF', 'PIT', 'TUB']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/NODDI_ICVF'), label_path=tree.get('warped_' + s + '_label_in_DTI_native'), threshold=threshold)
            idps[s + '_mean_intensity_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s + '_median_intensity_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s + '_max_intensity_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s + '_90th_percentile_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s + '_95th_percentile_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']
            idps[s + '_glm_mod=NODDI_ICVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = \
                runRegression(in_path=tree.get('data/NODDI_ICVF'), d_path=tree.get('warped_' + s + '_label_in_DTI_native'), o_path=tree.get('temp_GLM_output'))

# Calculate intensity IDPs in NODDI ISOVF
        anat_structs = ['AON', 'PIF', 'PIT', 'TUB']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/NODDI_ISOVF'), label_path=tree.get('warped_' + s + '_label_in_DTI_native'), threshold=threshold)
            idps[s + '_mean_intensity_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s + '_median_intensity_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s + '_max_intensity_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s + '_90th_percentile_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s + '_95th_percentile_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']
            idps[s + '_glm_mod=NODDI_ISOVF_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = \
                runRegression(in_path=tree.get('data/NODDI_ISOVF'), d_path=tree.get('warped_' + s + '_label_in_DTI_native'), o_path=tree.get('temp_GLM_output'))

# Calculate intensity IDPs in NODDI OD
        anat_structs = ['AON', 'PIF', 'PIT', 'TUB']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/NODDI_OD'), label_path=tree.get('warped_' + s + '_label_in_DTI_native'), threshold=threshold)
            idps[s + '_mean_intensity_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['mean']
            idps[s + '_median_intensity_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['median']
            idps[s + '_max_intensity_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['max']
            idps[s + '_90th_percentile_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['90th_percentile']
            idps[s + '_95th_percentile_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['95th_percentile']
            idps[s + '_glm_mod=NODDI_OD_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = \
                runRegression(in_path=tree.get('data/NODDI_OD'), d_path=tree.get('warped_' + s + '_label_in_DTI_native'), o_path=tree.get('temp_GLM_output'))

# Calculate intensity IDPs in T1_GM_mod in MNI space
        anat_structs = ['AON_T1_MNI', 'PIF_T1_MNI', 'PIT_T1_MNI', 'TUB_T1_MNI']
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/T1_GM_mod'), label_path=tree.get(s+'_label'), threshold=threshold)
            idps[s + '_sum_mod=T1_MNI_GM_mod_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['sum']
            idps[s + '_glm_mod=T1_MNI_GM_mod_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = \
                runRegression(in_path=tree.get('data/T1_GM_mod'), d_path=tree.get(s + '_label'), o_path=tree.get('temp_GLM_output'))

# Calculate intensity IDPs in T1_GM_mod in MNI space
    anat_structs = ['OC_MNI_2mm']
    thresholds = [0.3]
    for threshold in thresholds:
        for s in anat_structs:
            intensity_dict = calcIntensity(img_path=tree.get('data/T1_GM_mod'), label_path=tree.get(s+'_label'), threshold=threshold)
            idps[s + '_sum_mod=T1_MNI_GM_mod_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['sum']

# # Calculate intensity IDPs in T1_GM_mod in oxford-mm-0 space
#     if os.path.exists(tree.get('warped_T1_brain_pve1_mask')) and os.path.exists(tree.get('mmorf_jac')):
#         fslmaths(tree.get('warped_T1_brain_pve1_mask')).mul(tree.get('mmorf_jac')).run(tree.get('warped_T1brain_GM_mod_mask'))
#         anat_structs = ['OC']
#         thresholds = [0.3]
#         for threshold in thresholds:
#             for s in anat_structs:
#                 intensity_dict = calcIntensity(img_path=tree.get('warped_T1brain_GM_mod_mask'), label_path=tree.get('OC_OXMM_label'), threshold=threshold)
#                 idps[s + '_sum_mod=T1_GM_mod_oxmm0_threshold=' + str(threshold) + '_warpResolution=' + warp_resolution] = intensity_dict['sum']

    df = pd.DataFrame(data=idps, index=[0])
    df.to_csv(tree.get('subject_IDPs'), header=True, index=False, na_rep='nan')
    return 1


if __name__ == "__main__":
    identity_path = os.getenv('FSLDIR')+'/etc/flirtsch/ident.mat'
    mmorf_path = os.getenv('MMORFDIR')
    mmorf_run_cmd = 'singularity run --nv ' + mmorf_path
    mmorf_exec_cmd = 'singularity exec ' + mmorf_path

    flags_required = {
        'input_df': [('-idf', '--inputdf'),'<dir>'],
        'tree': [('-t', '--tree'),'<path>'],
        'output': [('-o', '--output'),'<dir>'],
        'template': [('-p', '--template'), '<dir>'],
        'modalities': [('-m', '--modalities'),'[T1_brain,T1_head,T1_head_and_neck,T2_head,T2_head_and_neck,DTI]',],
        'warpres': [('-w', '--warpres'), '[1mm,2mm]']
    }
    help_required = {
        'input_df': 'Directory containing the defaced subjects/timepoints',
        'tree': 'Path to FSL Filetree describing the subject-specific directory structure',
        'output': 'Output directory',
        'template': 'Template directory',
        'modalities': 'Choose between T1_head+T2_head+DTI and T1_head_and_neck+T2_head_and_neck+DTI',
        'warpres': 'Choose between 1mm or 2mm warp resolution'
    }

    flags_optional = {
        'subids': [('-s', '--subids'),'<path>'],
        'affine': [('-aff', '--affine'),'[True,False]'],
        'nonlinear': [('-nln', '--nonlinear'),'[True,False]']
    }
    help_optional = {
        'subids': 'Path to .csv file containing one subject ID per row: subject IDs have to indentify the sub-directories of the \'input\' argument (optional)'
                  'if not provided all sub-directories of the \'input\' argument will be used',
        'affine': 'Run affine template construction (required for affine)',
        'nonlinear': 'Run nonlinear template construction (required for nonlinear)'
    }

    parser = argparse.ArgumentParser(description='Atlas-based segmentation based on multimodal nonlinear registrations with T1+T2+DTI data.',
                                     usage='\npython runAnalysis_compact.py -idf /path/to/subjects/subjectsAll/ '
                                           '-t ./data_structure.tree -o /output/folder/ '
                                           '-p /path/to/oxford-mm-0/ '
                                           '-m T1_head,T2_head,DTI -s subject_IDs.csv -aff True -nln True -w 2mm')
    for key in flags_required.keys():
        parser.add_argument(*flags_required[key][0], help=help_required[key], metavar=flags_required[key][1], required=True)
    for key in flags_optional.keys():
        parser.add_argument(*flags_optional[key][0], help=help_optional[key], metavar=flags_optional[key][1])
    args = parser.parse_args()

    data_dir = args.inputdf
    tag = os.path.basename(os.path.abspath(args.output))
    base_dir = args.output
    temp_dir = args.template
    tree_path = args.tree
    modalities = args.modalities.split(',')
    warp_resolution = args.warpres

    if args.subids is not None:
        id_path = args.subids
        df_ids = pd.read_csv(id_path, header=None, names=['subject_ID'], dtype={'subject_ID': str})
        ls_ids = df_ids['subject_ID'].tolist()
    else:
        ls_ids = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        ls_ids.sort()

    clamping_on = True
    calcIDPs_on = True
    affine_on = args.affine == 'True'
    nln_on = args.nonlinear == 'True'

    job_ids = ['' for _ in range(100)]
    task_count = 0

    os.mkdir(base_dir, mode=0o750) if not os.path.exists(base_dir) else print(base_dir + ' exists')
    # os.chmod(base_dir,0o775)

    tree = FileTree.read(tree_path, top_level='')
    tree = tree.update(data_dir=data_dir,ukb_template_dir=temp_dir,cmore_dir=base_dir)
    script_dir = tree.get('script_dir')
    os.mkdir(script_dir,mode=0o750) if not os.path.exists(script_dir) else print(script_dir+' exists')
    log_dir =tree.get('log_dir')
    os.mkdir(log_dir,mode=0o750) if not os.path.exists(log_dir) else print(log_dir+' exists')
    misc_dir = tree.get('misc_dir')
    os.mkdir(misc_dir, mode=0o750) if not os.path.exists(misc_dir) else print(misc_dir + ' exists')
    shutil.copyfile(identity_path,tree.get('identity_mat'))

    cpu_queue = 'short.qc'
    gpu_queue = 'gpu8.q'

# Affine registrations
    if affine_on:

# Soft clamping of high skull intensities
        T1_clamped_path = tree.get('T1_head_clamped_nln_template')
        if clamping_on and not os.path.exists(T1_clamped_path):
            T1_head_path = tree.get('T1_head_nln_template')
            clampImage(T1_head_path, T1_clamped_path)

        task_name = '{:03d}_affinePart'.format(task_count)
        script_path = os.path.join(script_dir, task_name + '.sh')
        with open(script_path, 'w') as f:
            for id in ls_ids:
                tree = tree.update(sub_id=id)
                jobcmd = func_to_cmd(affinePart,
                                     args=(tree, clamping_on),
                                     tmp_dir=script_dir,
                                     kwargs=None,
                                     clean="never")
                jobcmd = jobcmd + '\n'
                f.write(jobcmd)

        job_ids[0] = fslsubWrapper(command=script_path, name=tag+'_'+task_name, log_dir=log_dir, queue=cpu_queue, wait_for=None, array_task=True)
        print('submitted: ' + task_name)

# Nonlinear registrations/transformations
    if nln_on:
        if not os.path.exists(tree.get('T1_brain_mask_weighted_nln_template')):
            fslmaths(tree.get('T1_brain_mask_nln_template')).bin().mul(7).add(1).inm(1).run(tree.get('T1_brain_mask_weighted_nln_template'))

        config_path = tree.get('mmorf_params',make_dir=True)
        if warp_resolution == '1mm':
            writeConfig1mm(modalities, config_path)
        elif warp_resolution == '2mm':
            writeConfig2mm(modalities,config_path)

        if 'T1_brain' in modalities:
            img_ref_T1brain_path = tree.get('T1_brain_nln_template')
        if 'T1_head' in modalities:
            if clamping_on:
                img_ref_T1head_path = tree.get('T1_head_clamped_nln_template')
            else:
                img_ref_T1head_path = tree.get('T1_head_nln_template')
        if 'T2_head' in modalities:
            img_ref_T2head_path = tree.get('T2_head_nln_template')
        if 'DTI' in modalities:
            img_ref_tensor_path = tree.get('DTI_tensor_nln_template')

        img_ref_T1brain_mask_path = tree.get('T1_brain_mask_weighted_nln_template')

        task_count += 1
        task_name = '{:03d}_nlnT_mmorf'.format(task_count)

# Nonlinear registration to template
        script_path = os.path.join(script_dir, task_name+'.sh')
        with open(script_path, 'w') as f:
            export_vars = {}
            for i,id in enumerate(ls_ids):
                tree = tree.update(sub_id=id)
                if clamping_on:
                    T1_head_path = tree.get('T1_head_clamped')
                else:
                    T1_head_path = tree.get('data/T1_head')

                if all(m in modalities for m in ['T1_head','T2_head','DTI']):
                    img_warp_space = img_ref_T1head_path
                    img_ref_scalar = [img_ref_T1head_path, img_ref_T2head_path]
                    img_mov_scalar = [T1_head_path,tree.get('data/T2_head')]
                    aff_ref_scalar = [tree.get('identity_mat'),tree.get('identity_mat')]
                    aff_mov_scalar = [tree.get('T1_brain_to_template_mat'),tree.get('T2_brain_to_template_mat')]
                    mask_ref_scalar = [img_ref_T1brain_mask_path,'NULL']
                    mask_mov_scalar = ['NULL','NULL']
                    img_ref_tensor = [img_ref_tensor_path]
                    img_mov_tensor = [tree.get('data/DTI_tensor')]
                    aff_ref_tensor = [tree.get('identity_mat')]
                    aff_mov_tensor = [tree.get('DTI_to_template_mat')]
                    mask_ref_tensor = [tree.get('T1_brain_mask_dti_nln_template')]
                    mask_mov_tensor = ['NULL']

                mmorf_script, export_var = mmorfWrapper(mmorf_run_cmd,config_path,img_warp_space=img_warp_space,
                                                img_ref_scalar=img_ref_scalar,img_mov_scalar=img_mov_scalar,
                                                aff_ref_scalar=aff_ref_scalar,aff_mov_scalar=aff_mov_scalar,
                                                mask_ref_scalar=mask_ref_scalar,mask_mov_scalar=mask_mov_scalar,
                                                img_ref_tensor=img_ref_tensor,img_mov_tensor=img_mov_tensor,
                                                aff_ref_tensor=aff_ref_tensor,aff_mov_tensor=aff_mov_tensor,
                                                mask_ref_tensor=mask_ref_tensor,mask_mov_tensor=mask_mov_tensor,
                                                warp_out=tree.get('mmorf_warp', make_dir=True),
                                                jac_det_out=tree.get('mmorf_jac'),
                                                bias_out=tree.get('mmorf_bias'))
                f.write(mmorf_script)

                for key, value in export_var.items():
                    if i == 0:
                        export_vars[key] = value
                    else:
                        export_vars[key] = export_vars[key]+value

            export_var_str = {}
            for key,value in export_vars.items():
                common_path = os.path.commonpath(value)
                export_var_str[key] = '"'+ key + '=' + ','.join([common_path]) + '"'

        job_ids[1] = fslsubWrapper(command=script_path, name=tag+'_'+task_name, log_dir=log_dir, queue=gpu_queue, wait_for=job_ids[0],
                                   array_task=True, coprocessor='cuda', coprocessor_class=None, coprocessor_multi="1", threads=1, export_var=[export_var_str['SINGULARITY_BIND']])
        print('submitted: ' + task_name)

# Inverse warps
        task_count += 1
        task_name = '{:03d}_nlnT_invert_warps'.format(task_count)
        script_path = os.path.join(script_dir, task_name + '.sh')
        with open(script_path, 'w') as f:
            for id in ls_ids:
                tree = tree.update(sub_id=id)
                jobcmd = func_to_cmd(createInverseWarps,
                                     args=(tree,),
                                     tmp_dir=script_dir,
                                     kwargs=None,
                                     clean="never")
                jobcmd = jobcmd + '\n'
                f.write(jobcmd)

        job_ids[2] = fslsubWrapper(command=script_path, name=tag+'_'+task_name, log_dir=log_dir, queue=cpu_queue, wait_for=job_ids[1], array_task=True)
        print('submitted: ' + task_name)

# Warp masks to subjects
        task_count += 1
        task_name = '{:03d}_nlnT_warpMasksToSubjects'.format(task_count)
        script_path = os.path.join(script_dir, task_name + '.sh')
        with open(script_path, 'w') as f:
            for id in ls_ids:
                tree = tree.update(sub_id=id)
                jobcmd = func_to_cmd(warpMasksToSubjects,
                                     args=(tree,),
                                     tmp_dir=script_dir,
                                     kwargs=None,
                                     clean="never")
                jobcmd = jobcmd + '\n'
                f.write(jobcmd)

        job_ids[3] = fslsubWrapper(command=script_path, name=tag+'_'+task_name, log_dir=log_dir, queue=cpu_queue, wait_for=job_ids[2], array_task=True)
        print('submitted: ' + task_name)

    if calcIDPs_on:
# Calculate IDPs
        task_count += 1
        task_name = '{:03d}_nlnT_calcIDPs'.format(task_count)
        script_path = os.path.join(script_dir, task_name + '.sh')
        with open(script_path, 'w') as f:
            for id in ls_ids:
                tree = tree.update(sub_id=id)
                jobcmd = func_to_cmd(extractIDPs,
                                     args=(tree, id, warp_resolution),
                                     tmp_dir=script_dir,
                                     kwargs=None,
                                     clean="never")
                jobcmd = jobcmd + '\n'
                f.write(jobcmd)

        job_ids[4] = fslsubWrapper(command=script_path, name=tag+'_'+task_name, log_dir=log_dir, queue=cpu_queue, wait_for=job_ids[3], array_task=True)
        print('submitted: ' + task_name)

# Concatenate all subject IDPs
        task_count += 1
        task_name = '{:03d}_nlnT_concatenateSubjects'.format(task_count)
        csv_paths = []
        for id in ls_ids:
            tree = tree.update(sub_id=id)
            csv_paths.append(tree.get('subject_IDPs'))
        output_path = tree.get('all_IDPs')
        jobcmd = func_to_cmd(concatVertically, args=(csv_paths, output_path), tmp_dir=script_dir, kwargs=None, clean="never")
        job_ids[5] = fslsubWrapper(command=jobcmd, name=tag+'_'+task_name, log_dir=log_dir, queue=cpu_queue, wait_for=job_ids[4], array_task=False)
        print('submitted: ' + task_name)

# Warp masks from MNI space to OX-MM space
        task_count += 1
        task_name = '{:03d}_nlnT_warpMNImasksToOXMM'.format(task_count)

        jobcmd = func_to_cmd(warpMasksToOXMM,
                             args=(tree,),
                             tmp_dir=script_dir,
                             kwargs=None,
                             clean="never")

        job_ids[6] = fslsubWrapper(command=jobcmd, name=tag+'_'+task_name, log_dir=log_dir, queue=cpu_queue, wait_for=job_ids[5], array_task=False)
        print('submitted: ' + task_name)




