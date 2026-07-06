import open3d as o3d
import numpy as np
import os
from plyfile import PlyData, PlyElement
from torch_points3d.modules.KPConv.plyutils import read_ply
import argparse

def read_point_cloud_with_attributes(file_path):
    """Read point cloud and its attributes"""
    pcd2 = o3d.io.read_point_cloud(file_path)
    pcd = read_ply(file_path)
    points = np.vstack((pcd['x'], pcd['y'], pcd['z'])).astype(np.float32).T
    
    # Example, here we assume these attributes are ordered, adjust according to actual situation
    # Extract corresponding attributes based on your point cloud file
    intensity = pcd["intensity"].astype(np.int64)
    semantic_seg = pcd["semantic_seg"].astype(np.int64)
    treeID = pcd["treeID"].astype(np.int64)
    # Create an index array
    indices = np.arange(len(points))

    return pcd2, points, intensity, semantic_seg, treeID, indices

def split_and_save_tiles(pcd, points, intensity, semantic_seg, treeID, indices, tile_size=100, overlap=5, base_dir="tiles"):
    """Split point cloud and its attributes into tiles, save tiles, and save the original indices of each point as a txt file"""
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # List to store the paths of all .ply output files
    output_files = []

    # Check if point cloud size is smaller than tile_size
    if (max_bound[0] - min_bound[0] <= tile_size and 
        max_bound[1] - min_bound[1] <= tile_size):
        
        # Save the entire point cloud as one tile
        data_struct = np.zeros(len(points), dtype=np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('semantic_seg', 'f4'), ('treeID', 'f4')]))
        data_struct['x'] = points[:,0]
        data_struct['y'] = points[:,1]
        data_struct['z'] = points[:,2]
        data_struct['intensity'] = intensity
        data_struct['semantic_seg'] = semantic_seg
        data_struct['treeID'] = treeID

        # Save the point cloud
        tile_filename = os.path.join(base_dir, "tile_0_0.ply")
        el = PlyElement.describe(data_struct, 'vertex', comments=['Created manually from las files.'])
        PlyData([el], byte_order='<').write(tile_filename)
        output_files.append(os.path.abspath(tile_filename))
        
        # Save indices as txt file
        indices_filename = os.path.join(base_dir, "tile_0_0_indices.txt")
        np.savetxt(indices_filename, indices, fmt='%d')

        # Write the list of .ply output files to a text file
        output_list_file = os.path.join(base_dir, "ply_output_file_paths.txt")
        with open(output_list_file, 'w') as f:
            for filepath in output_files:
                f.write(f"{filepath}\n")

        return  # Exit the function since there's no need to tile

    for i in range(int(np.ceil((max_bound[0] - min_bound[0]) / tile_size))):
        for j in range(int(np.ceil((max_bound[1] - min_bound[1]) / tile_size))):
            tile_min = [min_bound[0] + i * tile_size - overlap, min_bound[1] + j * tile_size - overlap, min_bound[2]]
            tile_max = [min_bound[0] + (i + 1) * tile_size + overlap, min_bound[1] + (j + 1) * tile_size + overlap, max_bound[2]]
            
            # Determine the points within the current tile
            in_tile = np.logical_and(np.all(points >= tile_min, axis=1), np.all(points < tile_max, axis=1))
            tile_points = points[in_tile]
            tile_intensity = intensity[in_tile]
            tile_semantic_seg = semantic_seg[in_tile]
            tile_treeID = treeID[in_tile]
            tile_indices = indices[in_tile]  # Retain the original indices of points in the current tile

            if len(tile_points) > 0:               
                data_struct = np.zeros(len(tile_points), dtype=np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('semantic_seg', 'f4'), ('treeID', 'f4')]))
                #data_struct = np.zeros(len, dtype=np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')]))
                data_struct['x'] = tile_points[:,0]
                data_struct['y'] = tile_points[:,1]
                data_struct['z'] = tile_points[:,2]
                data_struct['intensity'] = tile_intensity
                data_struct['semantic_seg'] = tile_semantic_seg
                data_struct['treeID'] = tile_treeID
                
                # Save tile point cloud
                tile_id = f"{i}_{j}"  # Create a unique ID based on the location
                tile_filename = os.path.join(base_dir, f"tile_{tile_id}.ply")
                #o3d.io.write_point_cloud(tile_filename, tile_pcd)
                el = PlyElement.describe(data_struct, 'vertex', comments=['Created manually from las files.'])
                PlyData([el], byte_order='<').write(tile_filename)
                output_files.append(os.path.abspath(tile_filename))
                
                # Save indices as txt file
                indices_filename = os.path.join(base_dir, f"tile_{tile_id}_indices.txt")
                np.savetxt(indices_filename, tile_indices, fmt='%d')

    # Write the list of .ply output files to a text file
    output_list_file = os.path.join(base_dir, "ply_output_file_paths.txt")
    with open(output_list_file, 'w') as f:
        for filepath in output_files:
            f.write(f"{filepath}\n")

def main():
    parser = argparse.ArgumentParser(description="Split large point cloud into tiles.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the large point cloud file.')
    parser.add_argument('--tile_size', type=int, default=50, help='Size of the tiles.')
    parser.add_argument('--overlap', type=int, default=5, help='Overlap size between tiles.')

    args = parser.parse_args()

    file_path = args.file_path
    tile_size = args.tile_size
    overlap = args.overlap

    pcd, points, intensity, semantic_seg, treeID, indices = read_point_cloud_with_attributes(file_path)
    
    # Create base_dir path with dynamic tile_size value
    directory = os.path.dirname(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    new_folder_name = f"tiles_{tile_size}_{file_name}"
    base_dir = os.path.join(directory, new_folder_name)

    split_and_save_tiles(pcd, points, intensity, semantic_seg, treeID, indices, tile_size=tile_size, overlap=overlap, base_dir=base_dir)


if __name__ == "__main__":
    main()
