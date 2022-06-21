import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
join = os.path.join


seg_path = r'../seg'
gt_path = r'../gt'
save_path = r'../result'
save_name = 'eval.csv'
filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
for i in range(1, 5):
    seg_metrics['DSC_{}'.format(i)] = list()
    seg_metrics['NSD-1mm_{}'.format(i)] = list()


for name in filenames:
    seg_metrics['Name'].append(name)
    # load grond truth and segmentation
    gt_nii = nb.load(join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = nb.load(join(seg_path, name)).get_fdata()

    for i in range(1, 5):
        if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
            DSC_i = 0
            NSD_i = 0
        else:
            surface_distances = compute_surface_distances(gt_data==i, seg_data==i, case_spacing)
            DSC_i = compute_dice_coefficient(gt_data==i, seg_data==i)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
        seg_metrics['DSC_{}'.format(i)].append(DSC_i)
        seg_metrics['NSD-1mm_{}'.format(i)].append(NSD_i)  

dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(join(save_path, save_name), index=False)
