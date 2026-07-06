import os
import shutil
import numpy as np
from plyfile import PlyData, PlyElement
import re
import argparse

def read_ply(file_path):
    """Read .ply file and return the data."""
    ply_data = PlyData.read(file_path)
    data = np.vstack([ply_data['vertex'][dim] for dim in ['x', 'y', 'z']]).T
    preds = ply_data['vertex']['preds']
    return data, preds

def read_ply_without_preds(file_path):
    """Read .ply file and return the data without predictions."""
    ply_data = PlyData.read(file_path)
    data = np.vstack([ply_data['vertex'][dim] for dim in ['x', 'y', 'z']]).T
    return data

def natural_sort_key(s):
    """Sort strings in human order"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def create_base_dir(base_dir):
    """Create base_dir and return its path"""
    #base_dir = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/outputs/tree_mix/all_tiles_outputs'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def collect_and_rename_files(src_dir, base_dir, ply_output_file):
    """Collect and rename files from source directories to base_dir"""
    timestamp_dirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))], key=natural_sort_key)

    # Read the ply_output_file to get the original tile file names
    with open(ply_output_file, 'r') as file:
        original_tile_files = [line.strip() for line in file.readlines()]

    tile_idx = 0
    
    for timestamp_dir in timestamp_dirs:
        timestamp_path = os.path.join(src_dir, timestamp_dir)
        
        instance_files = sorted([f for f in os.listdir(timestamp_path) if f.startswith('Instance_Results_forEval') and f.endswith('.ply')], key=natural_sort_key)
        semantic_files = sorted([f for f in os.listdir(timestamp_path) if f.startswith('Semantic_results_forEval') and f.endswith('.ply')], key=natural_sort_key)

        for instance_file, semantic_file in zip(instance_files, semantic_files):
            if tile_idx >= len(original_tile_files):
                break

            original_tile_file = original_tile_files[tile_idx]
            tile_id = os.path.splitext(os.path.basename(original_tile_file))[0].split('_')[-2] + '_' + os.path.splitext(os.path.basename(original_tile_file))[0].split('_')[-1]

            new_instance_file = f'Instance_Results_forEval_tile_{tile_id}.ply'
            new_semantic_file = f'Semantic_results_forEval_tile_{tile_id}.ply'
            index_file = f'tile_{tile_id}_indices.txt'

            # Copy prediction files
            shutil.copy(os.path.join(timestamp_path, instance_file), os.path.join(base_dir, new_instance_file))
            shutil.copy(os.path.join(timestamp_path, semantic_file), os.path.join(base_dir, new_semantic_file))

            # Copy index file
            original_index_file = os.path.join(os.path.dirname(ply_output_file), index_file)
            shutil.copy(original_index_file, os.path.join(base_dir, index_file))

            tile_idx += 1

def merge_tiles(base_dir, output_file, original_point_cloud_file, tile_size=100, overlap=5):
    """Merge tiles into a single point cloud based on original indices."""
    # Read the original point cloud
    original_points = read_ply_without_preds(original_point_cloud_file)
    num_points = len(original_points)

    # Create arrays to store the merged predictions
    merged_instance_preds = np.full(num_points, -1, dtype=np.int16)
    merged_semantic_preds = np.full(num_points, -1, dtype=np.int16)

    max_instance = 0

    # List all instance and semantic prediction files
    instance_files = sorted([f for f in os.listdir(base_dir) if f.startswith('Instance_Results_forEval_tile') and f.endswith('.ply')], key=natural_sort_key)
    semantic_files = sorted([f for f in os.listdir(base_dir) if f.startswith('Semantic_results_forEval_tile') and f.endswith('.ply')], key=natural_sort_key)

    for instance_file, semantic_file in zip(instance_files, semantic_files):
        tile_id = '_'.join(instance_file.split('_')[-2:]).replace('.ply', '')
        indices_file = os.path.join(base_dir, f"tile_{tile_id}_indices.txt")
        tile_indices = np.loadtxt(indices_file, dtype=int)

        instance_data, instance_preds = read_ply(os.path.join(base_dir, instance_file))
        _, semantic_preds = read_ply(os.path.join(base_dir, semantic_file))

        # Merge instance predictions
        all_pre_ins, max_instance = merge_instance_predictions(merged_instance_preds, instance_preds, tile_indices, max_instance)

        # Merge semantic predictions with priority to the first encountered prediction
        for idx, pred in zip(tile_indices, semantic_preds):
            if merged_semantic_preds[idx] == -1:
                merged_semantic_preds[idx] = pred

    # Filter out points where merged_semantic_preds is -1
    valid_points = merged_semantic_preds != -1

    # Create the final merged point cloud with instance and semantic predictions
    merged_data = np.zeros(np.sum(valid_points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                                        ('instance_preds', 'int16'), ('semantic_preds', 'int16')])
    merged_data['x'] = original_points[valid_points, 0]
    merged_data['y'] = original_points[valid_points, 1]
    merged_data['z'] = original_points[valid_points, 2]
    merged_data['instance_preds'] = merged_instance_preds[valid_points]
    merged_data['semantic_preds'] = merged_semantic_preds[valid_points]

    # Save the merged result
    el = PlyElement.describe(merged_data, 'vertex')
    PlyData([el], byte_order='<').write(output_file)
    
    '''# Create the final merged point cloud with instance and semantic predictions
    merged_data = np.zeros(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                              ('instance_preds', 'int16'), ('semantic_preds', 'int16')])
    merged_data['x'] = original_points[:, 0]
    merged_data['y'] = original_points[:, 1]
    merged_data['z'] = original_points[:, 2]
    merged_data['instance_preds'] = merged_instance_preds
    merged_data['semantic_preds'] = merged_semantic_preds

    # Save the merged result
    el = PlyElement.describe(merged_data, 'vertex')
    PlyData([el], byte_order='<').write(output_file)'''

def merge_single_tile(base_dir, output_file):
    """Merge a single tile into a single point cloud."""
    # Find the single instance and semantic prediction files
    instance_file = [f for f in os.listdir(base_dir) if f.startswith('Instance_Results_forEval_tile') and f.endswith('.ply')][0]
    semantic_file = [f for f in os.listdir(base_dir) if f.startswith('Semantic_results_forEval_tile') and f.endswith('.ply')][0]

    # Read the instance and semantic predictions
    instance_data, instance_preds = read_ply(os.path.join(base_dir, instance_file))
    _, semantic_preds = read_ply(os.path.join(base_dir, semantic_file))

    # Create the merged data
    merged_data = np.zeros(len(instance_data), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                                      ('instance_preds', 'int16'), ('semantic_preds', 'int16')])
    merged_data['x'] = instance_data[:, 0]
    merged_data['y'] = instance_data[:, 1]
    merged_data['z'] = instance_data[:, 2]
    merged_data['instance_preds'] = instance_preds
    merged_data['semantic_preds'] = semantic_preds

    # Save the merged result
    el = PlyElement.describe(merged_data, 'vertex')
    PlyData([el], byte_order='<').write(output_file)
    print(f"Processed single tile and saved to {output_file}.")

def merge_instance_predictions(all_pre_ins, pre_ins, originids, max_instance, iou_threshold=0.3):
    t_num_clusters = len(np.unique(pre_ins[pre_ins != -1]))
    mask_valid = pre_ins != -1

    # All points have no label
    if len(originids[mask_valid]) == 0:
        mask_valid = pre_ins != -1
        all_pre_ins[originids[mask_valid]] = pre_ins[mask_valid] + max_instance
        max_instance += t_num_clusters
    else:
        new_label = pre_ins.reshape(-1)
        for ii_idx in range(t_num_clusters):
            new_label_ii_idx = originids[np.argwhere(new_label == ii_idx).reshape(-1)]
            new_has_old_idx = new_label_ii_idx[np.argwhere(all_pre_ins[new_label_ii_idx] != -1)].reshape(-1)
            new_not_old_idx = new_label_ii_idx[np.argwhere(all_pre_ins[new_label_ii_idx] == -1)].reshape(-1)

            if len(new_has_old_idx) == 0:
                all_pre_ins[new_not_old_idx] = max_instance
                max_instance += 1
            elif len(new_not_old_idx) == 0:
                continue
            else:
                old_labels_ii = all_pre_ins[new_has_old_idx]
                unique_old_labels = np.unique(old_labels_ii)
                max_iou = 0
                max_iou_oldlabel = 0
                for g in unique_old_labels:
                    idx_old_all = originids[np.argwhere(all_pre_ins[originids] == g).reshape(-1)]
                    inter_label_idx = np.intersect1d(idx_old_all, new_label_ii_idx)
                    iou1 = float(len(inter_label_idx)) / float(len(idx_old_all))
                    iou2 = float(len(inter_label_idx)) / float(len(new_label_ii_idx))
                    iou = max(iou1, iou2)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_oldlabel = g

                if max_iou > iou_threshold:
                    all_pre_ins[new_not_old_idx] = max_iou_oldlabel
                else:
                    all_pre_ins[new_not_old_idx] = max_instance
                    max_instance += 1

    return all_pre_ins, max_instance

def main():
    '''src_dir = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/outputs/tree_mix/eval'
    ply_output_file = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/data_test/tiles_50_ULS_Åsmåsan_VUX1_2017_80_120mAGL/ply_output_file_paths.txt'
    base_dir = create_base_dir()
    output_file = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/outputs/tree_mix/merged_result.ply'
    original_point_cloud_file = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/data_test/ULS_Åsmåsan_VUX1_2017_80_120mAGL.ply'
    '''
    #src_dir = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/outputs/tree_mix/eval'
    #ply_output_file = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/data_set4classes/BlueCat/tiles_50_RN_merged_trees_panoptic/ply_output_file_paths.txt'
    #base_dir = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/outputs/tree_mix/all_tiles_outputs_RN_merged_trees_panoptic'
    #output_file = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/outputs/tree_mix/merged_result_new.ply'
    #original_point_cloud_file = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/data_set4classes/BlueCat/RN_merged_trees_panoptic.ply'


    parser = argparse.ArgumentParser(description="Merge tiles into a single point cloud.")
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory containing tile evaluation results.')
    parser.add_argument('--ply_output_file', type=str, required=True, help='Path to the ply output file paths.')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory to store merged results.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the final merged output file.')
    parser.add_argument('--original_point_cloud_file', type=str, required=True, help='Path to the original point cloud file.')
    parser.add_argument('--overlap', type=int, default=5, help='Overlap size between tiles.')

    args = parser.parse_args()

    src_dir = args.src_dir
    ply_output_file = args.ply_output_file
    base_dir = create_base_dir(args.base_dir)
    output_file = args.output_file
    original_point_cloud_file = args.original_point_cloud_file

    # Read the ply_output_file to check the number of tiles
    with open(ply_output_file, 'r') as file:
        tile_files = [line.strip() for line in file.readlines()]

    # Collect and rename files regardless of the number of tiles
    collect_and_rename_files(src_dir, base_dir, ply_output_file)

    if len(tile_files) == 1:
        # Only one tile, merge the single tile
        merge_single_tile(base_dir, output_file)
    else:
        # Multiple tiles, proceed with merging
        merge_tiles(base_dir, output_file, original_point_cloud_file, tile_size=100, overlap=args.overlap)
        #merge_tiles(base_dir, output_file, original_point_cloud_file, tile_size=100, overlap=5)


    #collect_and_rename_files(src_dir, base_dir, ply_output_file)
    #merge_tiles(base_dir, output_file, original_point_cloud_file, tile_size=100, overlap=5)

if __name__ == "__main__":
    main()
