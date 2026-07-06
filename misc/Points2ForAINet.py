'''
This script aims to adapt point clouds to be compatible with running ForAINet

'''

#!/usr/bin/env python3
import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
import argparse
import sys
import logging
import yaml
from typing import List, Optional, Dict, Tuple

# NEW: import tqdm
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    description = '''
Convert point cloud files between LAS/LAZ and PLY formats with offset management.

This tool preprocesses point clouds for ForAINet deep learning pipeline by:
  - Converting between formats (LAS, LAZ, PLY)
  - Subtracting coordinate offsets to ensure non-negative values
  - Restoring original coordinates after classification
'''

    epilog = '''
EXAMPLES:
  Preprocessing (before classification):
    # Calculate offsets and convert a single LAS file to PLY
    python Points2ForAINet.py mycloud.las --calculate-offsets

    # Process all LAS/PLY files in a directory
    python Points2ForAINet.py /data/pointclouds/ -e .las .ply --calculate-offsets

    # Process recursively with custom output directory
    python Points2ForAINet.py /data/ -r --calculate-offsets -o /data/preprocessed/

  Postprocessing (after classification):
    # Restore coordinates for a single dummy file
    python Points2ForAINet.py dummy_mycloud.ply --restore-offsets

    # Restore and convert to LAS format
    python Points2ForAINet.py dummy_mycloud.ply --restore-offsets --output-format las

    # Restore all dummy files in a directory
    python Points2ForAINet.py /data/results/ -e .ply --restore-offsets

  Other operations:
    # Convert PLY to LAZ without offset operations
    python Points2ForAINet.py mycloud.ply --output-format laz

    # Dry run to see what would be processed
    python Points2ForAINet.py /data/ --calculate-offsets --dry-run

    # Use custom offset file
    python Points2ForAINet.py dummy_mycloud.ply --restore-offsets --offset-file /path/to/offset.yml

WORKFLOW:
  1. Preprocessing:  original.las  -->  dummy_original.ply  (offsets saved to offset.yml)
  2. Run ForAINet classification on dummy_*.ply files
  3. Postprocessing: dummy_original.ply  -->  restored_original.ply  (coordinates restored)

OUTPUT FILES:
  - Preprocessing creates files with "dummy_" prefix
  - Postprocessing creates files with "restored_" prefix
  - Offset data is stored in offset.yml (YAML format)
'''

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_path',
        nargs='?',
        default='.',
        help='Input file or directory path (default: current directory)'
    )
    parser.add_argument(
        '-e', '--extensions',
        nargs='+',
        default=['.ply', '.las', '.laz'],
        help='File extensions to process (default: .ply .las .laz)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory (default: same as input)'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process files recursively in subdirectories'
    )
    parser.add_argument(
        '--output-format',
        choices=['ply', 'las', 'laz'],
        default='ply',
        help='Output format (default: ply)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually converting'
    )
    parser.add_argument(
        '--calculate-offsets',
        action='store_true',
        help='Calculate and apply coordinate offsets (preprocessing mode)'
    )
    parser.add_argument(
        '--restore-offsets',
        action='store_true',
        help='Restore original coordinates from offsets (postprocessing mode)'
    )
    parser.add_argument(
        '--offset-file',
        help='Path to offset YAML file (default: offset.yml in script directory or input directory)'
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.calculate_offsets and args.restore_offsets:
        logger.error("Cannot use both --calculate-offsets and --restore-offsets simultaneously")
        sys.exit(1)
    
    input_path = Path(args.input_path)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Normalize extensions
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                  for ext in args.extensions]
    logger.info(f"Processing files with extensions: {extensions}")

    # Find files to process
    if input_path.is_file():
        files_to_process = [input_path] if input_path.suffix.lower() in extensions else []
    else:
        files_to_process = find_files(input_path, extensions, args.recursive)

    n_files = len(files_to_process)
    if not files_to_process:
        logger.warning(f"No files found with extensions {extensions} in {input_path}")
        return

    # Print total number of files to process
    print(f"Number of files to process: {n_files}")
    logger.info(f"Found {n_files} files to process")

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent if input_path.is_file() else input_path

    # Determine offset file path
    if args.offset_file:
        offset_file_path = Path(args.offset_file)
    else:
        # Look for offset.yml in script directory first, then in input directory
        script_dir = Path(__file__).parent
        offset_file_path = script_dir / "offset.yml"
        if not offset_file_path.exists():
            offset_file_path = (input_path.parent if input_path.is_file() else input_path) / "offset.yml"

    if args.dry_run:
        logger.info("DRY RUN - Files that would be processed:")
        for file_path in files_to_process:
            output_path = get_output_path(file_path, output_dir, args.output_format, args.restore_offsets)
            logger.info(f"  {file_path} -> {output_path}")
        if args.calculate_offsets:
            logger.info(f"Offset file would be saved to: {offset_file_path}")
        elif args.restore_offsets:
            logger.info(f"Offset file would be read from: {offset_file_path}")
        return

    # Handle offset operations
    offsets_data = {}
    
    if args.calculate_offsets:
        logger.info("Calculating coordinate offsets...")
        offsets_data = calculate_offsets_for_files(files_to_process)
        save_offsets(offsets_data, offset_file_path)
        logger.info(f"Offsets saved to: {offset_file_path}")
    
    elif args.restore_offsets:
        logger.info("Loading coordinate offsets...")
        if not offset_file_path.exists():
            logger.error(f"Offset file not found: {offset_file_path}")
            sys.exit(1)
        offsets_data = load_offsets(offset_file_path)
        logger.info(f"Offsets loaded from: {offset_file_path}")

    # Process files with a progress bar
    success_count = 0
    error_count = 0

    for file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        try:
            output_path = get_output_path(file_path, output_dir, args.output_format, args.restore_offsets)

            if output_path.exists() and not args.overwrite:
                logger.warning(f"Output file exists, skipping: {output_path}")
                continue

            logger.info(f"Processing: {file_path}")
            
            # Get offset for this file if needed
            file_offset = None
            if args.calculate_offsets or args.restore_offsets:
                # Try multiple keys to find the offset
                possible_keys = [
                    str(file_path.resolve()),  # Full resolved path
                    str(file_path),            # Relative path
                    str(file_path.absolute())  # Absolute path
                ]
                
                # For restore mode, also try to find the original file path
                if args.restore_offsets and file_path.name.startswith("dummy_"):
                    original_name = file_path.name[6:]  # Remove "dummy_" prefix
                    original_path = file_path.parent / original_name
                    possible_keys.extend([
                        str(original_path.resolve()),
                        str(original_path),
                        str(original_path.absolute())
                    ])
                
                # Try each possible key
                found_key = None
                for key in possible_keys:
                    if key in offsets_data:
                        file_offset = offsets_data[key]
                        found_key = key
                        break
                
                if file_offset is None:
                    logger.warning(f"No offset data found for {file_path}")
                    logger.debug(f"Tried keys: {possible_keys}")
                    logger.debug(f"Available keys in offset file: {list(offsets_data.keys())[:5]}...")  # Show first 5 keys
                else:
                    logger.info(f"Found offset for {file_path} using key: {found_key}")
                    logger.debug(f"Offset data: {file_offset}")
            
            convert_file(file_path, output_path, file_offset, args.restore_offsets)
            logger.info(f"Successfully converted: {file_path} -> {output_path}")
            success_count += 1

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            error_count += 1

    logger.info(f"Conversion complete. Success: {success_count}, Errors: {error_count}")

def find_files(directory: Path, extensions: List[str], recursive: bool = False) -> List[Path]:
    files = []
    if recursive:
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
    return sorted(files)

def get_output_path(input_path: Path, output_dir: Path, output_format: str, restore_mode: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_name = input_path.stem
    
    # Choose prefix based on mode
    if restore_mode:
        # Always use "restored_" prefix and remove any existing prefix when restoring
        if original_name.startswith("dummy_"):
            original_name = original_name[6:]  # Remove "dummy_" prefix
        elif original_name.startswith("restored_"):
            original_name = original_name[9:]  # Remove "restored_" prefix
        prefix = "restored_"
    else:
        prefix = "dummy_"
    
    if output_format == 'ply':
        output_filename = f"{prefix}{original_name}.ply"
    elif output_format == 'las':
        output_filename = f"{prefix}{original_name}.las"
    elif output_format == 'laz':
        output_filename = f"{prefix}{original_name}.laz"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return output_dir / output_filename

def calculate_offsets_for_files(files: List[Path]) -> Dict[str, Dict[str, float]]:
    """Calculate offsets for all input files"""
    offsets_data = {}
    
    for file_path in tqdm(files, desc="Calculating offsets", unit="file"):
        try:
            offset_info = calculate_single_file_offset(file_path)
            
            # Store offset with multiple keys to handle both input and output paths
            input_key = str(file_path.resolve())
            offsets_data[input_key] = offset_info
            
            # Also store with the expected dummy output path as key
            # This allows restore mode to find the offset using the dummy file path
            output_path = get_dummy_output_path(file_path)
            dummy_key = str(output_path.resolve())
            offsets_data[dummy_key] = offset_info
            
            logger.debug(f"Calculated offset for {file_path}: {offset_info}")
            logger.debug(f"Stored offset with keys: {input_key} and {dummy_key}")
        except Exception as e:
            logger.error(f"Error calculating offset for {file_path}: {e}")
    
    return offsets_data

def get_dummy_output_path(input_path: Path) -> Path:
    """Get the expected dummy output path for a given input path"""
    # This mimics the logic in get_output_path for preprocessing mode
    original_name = input_path.stem
    dummy_filename = f"dummy_{original_name}.ply"
    return input_path.parent / dummy_filename

def calculate_single_file_offset(file_path: Path) -> Dict[str, float]:
    """Calculate offset for a single file ensuring positive coordinates after offset"""
    input_ext = file_path.suffix.lower()
    
    if input_ext in [".laz", ".las"]:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).astype(np.float64).T
    elif input_ext == ".ply":
        data = PlyData.read(file_path)
        vertex_data = data['vertex'].data
        points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).astype(np.float64).T
    else:
        raise ValueError(f"Unsupported file type: {input_ext}")
    
    # Use minimum values as offsets to ensure all coordinates become >= 0
    # This guarantees positive coordinates after preprocessing
    min_x = float(np.min(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    min_z = float(np.min(points[:, 2]))
    
    # Store additional statistics for validation
    return {
        'offset_x': min_x,
        'offset_y': min_y, 
        'offset_z': min_z,
        'original_range_x': float(np.ptp(points[:, 0])),  # peak-to-peak range
        'original_range_y': float(np.ptp(points[:, 1])),
        'original_range_z': float(np.ptp(points[:, 2])),
        'original_min_x': min_x,
        'original_min_y': min_y,
        'original_min_z': min_z,
        'n_points': len(points)
    }

def save_offsets(offsets_data: Dict[str, Dict[str, float]], offset_file_path: Path):
    """Save offsets to YAML file"""
    offset_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(offset_file_path, 'w') as f:
        yaml.dump(offsets_data, f, default_flow_style=False)

def load_offsets(offset_file_path: Path) -> Dict[str, Dict[str, float]]:
    """Load offsets from YAML file"""
    with open(offset_file_path, 'r') as f:
        return yaml.safe_load(f)

def get_dynamic_dtype_fields(input_path: Path) -> List[Tuple[str, str]]:
    """Dynamically determine dtype fields from input file"""
    input_ext = input_path.suffix.lower()
    
    if input_ext in [".laz", ".las"]:
        las = laspy.read(input_path)
        # Start with basic coordinate fields
        fields = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        
        # Add intensity if available
        if hasattr(las, 'intensity'):
            fields.append(('intensity', 'f8'))
        
        # Add other common LAS fields
        for field_name in ['red', 'green', 'blue', 'classification', 'return_number', 'number_of_returns']:
            if hasattr(las, field_name):
                # Determine appropriate dtype based on field
                if field_name in ['red', 'green', 'blue', 'intensity']:
                    fields.append((field_name, 'u2'))  # uint16
                elif field_name in ['classification', 'return_number', 'number_of_returns']:
                    fields.append((field_name, 'u1'))  # uint8
                else:
                    fields.append((field_name, 'f8'))  # default to float64
        
    elif input_ext == ".ply":
        data = PlyData.read(input_path)
        vertex_data = data['vertex'].data
        fields = []
        
        # Convert numpy dtypes to our format
        for name in vertex_data.dtype.names:
            numpy_dtype = vertex_data.dtype.fields[name][0]
            
            # Map numpy dtypes to our format
            if numpy_dtype in [np.float32, np.float64]:
                fields.append((name, 'f8'))
            elif numpy_dtype in [np.uint8]:
                fields.append((name, 'u1'))
            elif numpy_dtype in [np.uint16]:
                fields.append((name, 'u2'))
            elif numpy_dtype in [np.uint32]:
                fields.append((name, 'u4'))
            elif numpy_dtype in [np.int8]:
                fields.append((name, 'i1'))
            elif numpy_dtype in [np.int16]:
                fields.append((name, 'i2'))
            elif numpy_dtype in [np.int32]:
                fields.append((name, 'i4'))
            else:
                # Default to float64 for unknown types
                fields.append((name, 'f8'))
                logger.warning(f"Unknown dtype {numpy_dtype} for field {name}, using f8")
    
    else:
        raise ValueError(f"Unsupported file type: {input_ext}")
    
    logger.debug(f"Dynamic fields for {input_path}: {fields}")
    return fields

def convert_file(input_path: Path, output_path: Path, offset: Optional[Dict[str, float]] = None, restore_mode: bool = False):
    # Get dynamic dtype fields based on input file
    dtype_fields = get_dynamic_dtype_fields(input_path)
    
    # In restore mode, we need to ensure we have the basic coordinate fields
    if restore_mode:
        # Ensure x, y, z are present and in the right format for coordinate restoration
        required_coords = ['x', 'y', 'z']
        field_names = [field[0] for field in dtype_fields]
        for coord in required_coords:
            if coord not in field_names:
                logger.error(f"Missing required coordinate field '{coord}' for restoration")
                raise ValueError(f"Cannot restore coordinates: missing field '{coord}'")
    else:
        # In preprocessing mode, use default fields plus dummy fields
        dtype_fields = [
            ('x', 'f8'),
            ('y', 'f8'),
            ('z', 'f8'),
            ('intensity', 'f8'),
            ('semantic_seg', 'u1'),  # dummy label
            ('treeID', 'u4')         # dummy tree ID
        ]
    
    input_ext = input_path.suffix.lower()
    output_ext = output_path.suffix.lower()
    
    if input_ext in [".laz", ".las"]:
        array = get_las(input_path, dtype_fields, offset, restore_mode)
    elif input_ext == ".ply":
        array = get_ply(input_path, dtype_fields, offset, restore_mode)
    else:
        raise ValueError(f"Unsupported input file type: {input_ext}")
    
    if output_ext == ".ply":
        array_to_ply(array, output_path)
    elif output_ext in [".las", ".laz"]:
        array_to_las(array, output_path)
    else:
        raise ValueError(f"Unsupported output file type: {output_ext}")

def apply_coordinate_transform(points: np.ndarray, offset: Dict[str, float], restore_mode: bool) -> np.ndarray:
    """Apply or restore coordinate transformation ensuring positive coordinates"""
    if offset is None:
        logger.warning("No offset provided for coordinate transformation")
        return points
    
    offset_x = offset['offset_x']
    offset_y = offset['offset_y'] 
    offset_z = offset['offset_z']
    
    # Log coordinate ranges before transformation
    before_range_x = np.ptp(points[:, 0])
    before_range_y = np.ptp(points[:, 1])
    before_range_z = np.ptp(points[:, 2])
    before_min_x = np.min(points[:, 0])
    before_min_y = np.min(points[:, 1])
    before_min_z = np.min(points[:, 2])
    
    logger.debug(f"Before transform - Min: x={before_min_x:.6f}, y={before_min_y:.6f}, z={before_min_z:.6f}")
    logger.debug(f"Before transform - Range: x={before_range_x:.6f}, y={before_range_y:.6f}, z={before_range_z:.6f}")
    
    if restore_mode:
        # Restore original coordinates by adding back the offsets
        logger.info(f"RESTORING coordinates: adding offsets x+{offset_x:.6f}, y+{offset_y:.6f}, z+{offset_z:.6f}")
        points[:, 0] = points[:, 0] + offset_x
        points[:, 1] = points[:, 1] + offset_y
        points[:, 2] = points[:, 2] + offset_z
        
        # Log results after restoration
        after_min_x = np.min(points[:, 0])
        after_min_y = np.min(points[:, 1])
        after_min_z = np.min(points[:, 2])
        logger.info(f"After restoration - Min: x={after_min_x:.6f}, y={after_min_y:.6f}, z={after_min_z:.6f}")
        
        # Validate restoration if we have original minimum values
        if 'original_min_x' in offset:
            expected_min_x = offset['original_min_x']
            expected_min_y = offset['original_min_y']
            expected_min_z = offset['original_min_z']
            logger.info(f"Expected mins: x={expected_min_x:.6f}, y={expected_min_y:.6f}, z={expected_min_z:.6f}")
            
            if abs(after_min_x - expected_min_x) > 1e-6:
                logger.error(f"X minimum mismatch after restoration: expected {expected_min_x:.6f}, got {after_min_x:.6f}")
            if abs(after_min_y - expected_min_y) > 1e-6:
                logger.error(f"Y minimum mismatch after restoration: expected {expected_min_y:.6f}, got {after_min_y:.6f}")
            if abs(after_min_z - expected_min_z) > 1e-6:
                logger.error(f"Z minimum mismatch after restoration: expected {expected_min_z:.6f}, got {after_min_z:.6f}")
        
        # Validate ranges
        if 'original_range_x' in offset:
            current_range_x = np.ptp(points[:, 0])
            expected_range_x = offset['original_range_x']
            if abs(current_range_x - expected_range_x) > 1e-6:
                logger.warning(f"X range mismatch after restoration: expected {expected_range_x:.6f}, got {current_range_x:.6f}")
    else:
        # Apply offset (subtract minimums) - this guarantees all coordinates >= 0
        logger.info(f"APPLYING offsets: subtracting x-{offset_x:.6f}, y-{offset_y:.6f}, z-{offset_z:.6f}")
        points[:, 0] = points[:, 0] - offset_x
        points[:, 1] = points[:, 1] - offset_y  
        points[:, 2] = points[:, 2] - offset_z
        
        # Verify all coordinates are non-negative (should be guaranteed now)
        min_x, min_y, min_z = np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])
        logger.info(f"After offset application - Min coordinates: x={min_x:.6f}, y={min_y:.6f}, z={min_z:.6f}")
        
        # This should never happen with min-based offsets, but let's check anyway
        if min_x < -1e-10 or min_y < -1e-10 or min_z < -1e-10:  # Allow for small floating point errors
            logger.error(f"Unexpected negative coordinates after offset: min_x={min_x:.10f}, min_y={min_y:.10f}, min_z={min_z:.10f}")
    
    return points

def get_las(input_path: Path, dtype_fields, offset: Optional[Dict[str, float]] = None, restore_mode: bool = False) -> np.ndarray:
    logger.debug(f"Reading LAS/LAZ file: {input_path}")
    las = laspy.read(input_path)
    
    # Get coordinates as float64 for precision
    points = np.vstack((las.x, las.y, las.z)).astype(np.float64).T
    
    # Apply coordinate transformation if offset is provided
    if offset is not None:
        points = apply_coordinate_transform(points, offset, restore_mode)
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    n_points = len(x)
    logger.debug(f"Loaded {n_points} points from LAS/LAZ file")
    
    # Create array with dynamic fields
    array = np.empty(n_points, dtype=dtype_fields)
    array['x'] = x
    array['y'] = y
    array['z'] = z
    
    # Fill other fields dynamically
    for field_name, field_type in dtype_fields:
        if field_name in ['x', 'y', 'z']:
            continue  # Already handled
        elif field_name == 'intensity':
            array[field_name] = getattr(las, 'intensity', np.zeros(n_points, dtype='f8'))
        elif hasattr(las, field_name):
            array[field_name] = getattr(las, field_name)
        else:
            # Set default values for missing fields
            if field_type in ['u1', 'i1']:
                array[field_name] = np.zeros(n_points, dtype='u1')
            elif field_type in ['u2', 'i2']:
                array[field_name] = np.zeros(n_points, dtype='u2')
            elif field_type in ['u4', 'i4']:
                array[field_name] = np.zeros(n_points, dtype='u4')
            else:
                array[field_name] = np.zeros(n_points, dtype='f8')
    
    return array

def get_ply(input_path: Path, dtype_fields, offset: Optional[Dict[str, float]] = None, restore_mode: bool = False) -> np.ndarray:
    logger.debug(f"Reading PLY file: {input_path}")
    data = PlyData.read(input_path)
    vertex_data = data['vertex'].data
    
    # Get coordinates as float64 for precision
    points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).astype(np.float64).T
    
    # Apply coordinate transformation if offset is provided
    if offset is not None:
        points = apply_coordinate_transform(points, offset, restore_mode)
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    n_points = len(x)
    logger.debug(f"Loaded {n_points} points from PLY file")
    
    # Create array with dynamic fields
    array = np.empty(n_points, dtype=dtype_fields)
    array['x'] = x
    array['y'] = y  
    array['z'] = z
    
    # Fill other fields dynamically
    for field_name, field_type in dtype_fields:
        if field_name in ['x', 'y', 'z']:
            continue  # Already handled
        elif field_name in vertex_data.dtype.names:
            # Field exists in PLY data
            array[field_name] = vertex_data[field_name]
        else:
            # Handle missing fields with appropriate defaults
            if field_name == 'intensity':
                # Look for intensity variants
                intensity_fields = ['intensity', 'scalar_Intensity', 'scalar_intensity', 'Intensity']
                found_intensity = False
                for intensity_field in intensity_fields:
                    if intensity_field in vertex_data.dtype.names:
                        array[field_name] = vertex_data[intensity_field]
                        found_intensity = True
                        break
                if not found_intensity:
                    array[field_name] = np.zeros(n_points, dtype='f8')
            else:
                # Set default values for missing fields
                if field_type in ['u1', 'i1']:
                    array[field_name] = np.zeros(n_points, dtype='u1')
                elif field_type in ['u2', 'i2']:
                    array[field_name] = np.zeros(n_points, dtype='u2')
                elif field_type in ['u4', 'i4']:
                    array[field_name] = np.zeros(n_points, dtype='u4')
                else:
                    array[field_name] = np.zeros(n_points, dtype='f8')
    
    return array

def array_to_ply(array: np.ndarray, output_path: Path):
    logger.debug(f"Writing PLY file: {output_path}")
    
    # Preserve all fields from the input array
    element = PlyElement.describe(array, 'vertex')
    PlyData([element], text=False).write(output_path)

def array_to_las(array: np.ndarray, output_path: Path):
    logger.debug(f"Writing LAS/LAZ file: {output_path}")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([np.min(array['x']), np.min(array['y']), np.min(array['z'])])
    
    las = laspy.LasData(header)
    las.x = array['x']
    las.y = array['y']
    las.z = array['z']
    
    # Handle other fields dynamically
    if 'intensity' in array.dtype.names:
        las.intensity = array['intensity'].astype(np.uint16)
    
    # Note: LAS format has limited field support compared to PLY
    # Additional fields from PLY may be lost when converting to LAS
    if len([name for name in array.dtype.names if name not in ['x', 'y', 'z', 'intensity']]) > 0:
        logger.warning("Some fields may be lost when converting to LAS format")
    
    las.write(output_path)

if __name__ == "__main__":
    main()