import h5py
import numpy as np
import sys

def view_h5_file(file_path):
    """
    View the contents of an H5 file, including groups, datasets, and attributes.
    
    Parameters:
    -----------
    file_path : str
        Path to the H5 file
    """
    # Open the file in read-only mode
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("\nFile Structure:")
        
        # Define a function to recursively explore the HDF5 file structure
        def print_structure(name, obj):
            indent = '    ' * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
                # Print attributes of the group
                if len(obj.attrs) > 0:
                    print(f"{indent}  Attributes:")
                    for key, val in obj.attrs.items():
                        print(f"{indent}    {key}: {val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}")
                print(f"{indent}  Shape: {obj.shape}")
                print(f"{indent}  Type: {obj.dtype}")
                # Print a sample of data for small datasets
                if np.prod(obj.shape) < 10:
                    print(f"{indent}  Data: {obj[...]}")
                elif len(obj.shape) == 1 and obj.shape[0] < 100:
                    print(f"{indent}  First 5 elements: {obj[0:5]}")
                # Print attributes of the dataset
                if len(obj.attrs) > 0:
                    print(f"{indent}  Attributes:")
                    for key, val in obj.attrs.items():
                        print(f"{indent}    {key}: {val}")
        
        # Visit all groups and datasets in the file
        f.visititems(print_structure)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_h5.py <h5_file_path>")
        sys.exit(1)
    
    view_h5_file(sys.argv[1])