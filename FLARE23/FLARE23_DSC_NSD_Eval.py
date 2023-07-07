#!/usr/bin/env python
import sys
import os
import nibabel as nb
import numpy as np
import glob
import gc
from collections import OrderedDict
from helpers.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from helpers.utils import time_elapsed

# from scipy.ndimage.measurements import label as label_connected_components
# from helpers.calc_metric import (dice,
#                                  detect_lesions,
#                                  compute_segmentation_scores,
#                                  compute_tumor_burden,
#                                  LARGE)


def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.
    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int
    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) ==1, print('mask label error!')
    z_index = np.where(organ_mask>0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper


# Check input directories.
submit_dir = os.path.join(sys.argv[1], 'res')
truth_dir = os.path.join(sys.argv[1], 'ref')
if not os.path.isdir(submit_dir):
    print("submit_dir {} doesn't exist".format(submit_dir))
    sys.exit()
if not os.path.isdir(truth_dir):
    print("truth_dir {} doesn't exist".format(submit_dir))
    sys.exit()

# Create output directory.
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------------- flare metrics
seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
label_tolerance = OrderedDict({'Liver': 5, 'RK':3, 'Spleen':3, 'Pancreas':5, 
                   'Aorta': 2, 'IVC':2, 'RAG':2, 'LAG':2, 'Gallbladder': 2,
                   'Esophagus':3, 'Stomach': 5, 'Duodenum': 7, 'LK':3, 'Tumor':2})
for organ in label_tolerance.keys():
    seg_metrics['{}_DSC'.format(organ)] = list()
for organ in label_tolerance.keys():
    seg_metrics['{}_NSD'.format(organ)] = list()
# -------------------------- flare metrics

# Iterate over all volumes in the reference list.
reference_volume_list = sorted(glob.glob(truth_dir+'/*.nii.gz'))
for reference_volume_fn in reference_volume_list:
    print("Starting with volume {}".format(reference_volume_fn))
    submission_volume_path = os.path.join(submit_dir, os.path.basename(reference_volume_fn))
    if not os.path.exists(submission_volume_path):
        raise ValueError("Submission volume not found - terminating!\n"
                         "Missing volume: {}".format(submission_volume_path))
    print("Found corresponding submission file {} for reference file {}"
          "".format(reference_volume_fn, submission_volume_path))
    t = time_elapsed()

    # Load reference and submission volumes with Nibabel.
    reference_volume = nb.load(reference_volume_fn)
    submission_volume = nb.load(submission_volume_path)

    # Get the current voxel spacing.
    voxel_spacing = reference_volume.header.get_zooms()[:3]

    # Get Numpy data and compress to int8.
    reference_volume = (reference_volume.get_fdata()).astype(np.int8)
    submission_volume = (submission_volume.get_fdata()).astype(np.int8)
    
    # Ensure that the shapes of the masks match.
    if submission_volume.shape!=reference_volume.shape:
        raise AttributeError("Shapes do not match! Prediction mask {}, "
                             "ground truth mask {}"
                             "".format(submission_volume.shape,
                                       reference_volume.shape))
    print("Done loading files ({:.2f} seconds)".format(t()))

    # ----------------------- flare metrics
    seg_metrics['Name'].append(os.path.basename(reference_volume_fn))
    for i, organ in enumerate(label_tolerance.keys(),1):
        if np.sum(reference_volume==i)==0 and np.sum(submission_volume==i)==0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(reference_volume==i)==0 and np.sum(submission_volume==i)>0:
            DSC_i = 0
            NSD_i = 0
        elif np.sum(reference_volume==i)>0 and np.sum(submission_volume==i)==0:  
            DSC_i = 0
            NSD_i = 0
        else:
            if i==5 or i==6 or i==10: # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
                z_lower, z_upper = find_lower_upper_zbound(reference_volume==i)
                organ_i_gt, organ_i_seg = reference_volume[:,:,z_lower:z_upper]==i, submission_volume[:,:,z_lower:z_upper]==i
            else:
                organ_i_gt, organ_i_seg = reference_volume==i, submission_volume==i
            DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
            if DSC_i < 0.1:
                NSD_i = 0
            else:
                surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, voxel_spacing)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
        seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))
        seg_metrics['{}_NSD'.format(organ)].append(round(NSD_i, 4))  
        # print(name, organ, round(DSC_i,4), 'tol:', label_tolerance[organ], round(NSD_i,4))
    # ----------------------- flare metrics

    
    print("Done processing volume (total time: {:.2f} seconds)"
          "".format(t.total_elapsed()))
    gc.collect()

overall_metrics = {}
for key, value in seg_metrics.items():
    if 'Name' not in key:
        overall_metrics[key] = round(np.mean(value), 4)

organ_dsc = []
organ_nsd = []
for key, value in overall_metrics.items():
    if 'Tumor' not in key:
        if 'DSC' in key:
            organ_dsc.append(value)
        if 'NSD' in key:
            organ_nsd.append(value)
overall_metrics['Organ_DSC'] = round(np.mean(organ_dsc), 4)
overall_metrics['Organ_NSD'] = round(np.mean(organ_nsd), 4)

print("Computed metrics:")
for key, value in overall_metrics.items():
    print("{}: {:.4f}".format(key, float(value)))

# Write metrics to file.
output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'w')
for key, value in overall_metrics.items():
    output_file.write("{}: {:.4f}\n".format(key, float(value)))
output_file.close()
