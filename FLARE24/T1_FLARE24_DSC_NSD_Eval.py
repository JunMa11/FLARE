#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on March 2024

@author: junma
"""

import numpy as np
import nibabel as nb
import os
join = os.path.join
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient


seg_path = 'path to segmentation'
gt_path = 'path to ground truth'
save_path = 'path to Results'
save_name = 'DSC_NSD_teamname.csv'
filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames.sort()

tumor_tolerance = 2
seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
seg_metrics['Lesion_DSC'] = list()
seg_metrics['Lesion_NSD'] = list()

for name in filenames:
    seg_metrics['Name'].append(name)
    # load grond truth and segmentation
    gt_nii = nb.load(join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

    if np.max(seg_data) > 1: 
        DSC_i = 0
        NSD_i = 0
    elif np.sum(gt_data)==0 and np.sum(seg_data)==0:
        DSC_i = 1
        NSD_i = 1
    elif np.sum(gt_data)==0 and np.sum(seg_data)>0:
        DSC_i = 0
        NSD_i = 0
    elif np.sum(gt_data)>0 and np.sum(seg_data)==0:  
        DSC_i = 0
        NSD_i = 0
    else:
        DSC_i = compute_dice_coefficient(gt_data, seg_data)
        if DSC_i < 0.2: # don't compute NSD if DSC is too low
            NSD_i = 0
        else:
            surface_distances = compute_surface_distances(gt_data, seg_data, case_spacing)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, tumor_tolerance)
    seg_metrics['Lesion_DSC'].append(round(DSC_i, 4))
    seg_metrics['Lesion_NSD'].append(round(NSD_i, 4))  
    print(name, 'DSC:', round(DSC_i, 4), 'NSD (tol: 2)', round(NSD_i, 4))

dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(join(save_path, save_name), index=False)

