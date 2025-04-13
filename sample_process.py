import h5py
import numpy as np
import os

def get_available_filename(base_path, ext):
    """
    Generate a filename that doesn't exist by adding incremental digits.
    Returns the available filename.
    """

    
    i = 1
    while True:
        # Use format specifier for consistent leading zeros
        new_name = f"{base_path}_{i:02d}{ext}"
        
        if not os.path.exists(new_name):
            return new_name
        i += 1

def save_files():
    with h5py.File('/home/rotogni/diffusion-lagr/datasets/trajectories_normalized_cond_60.h5', 'r') as h5f:
        rx0 = np.array(h5f.get('min'))
        rx1 = np.array(h5f.get('max'))
    # Generate filenames that don't exist
    pos_base_path = '/home/rotogni/diffusion-lagr/pos_samples_256x60x3_cond_60_att'
    init_base_path = '/home/rotogni/diffusion-lagr/init_samples_256x10x3_cond_60_att'

    pos_filename = get_available_filename(pos_base_path, '.npy')
    init_filename = get_available_filename(init_base_path, '.npy')

    pos = (np.load('/home/rotogni/diffusion-lagr/samples_256x60x3.npz')['arr_0']+1)*(rx1-rx0)/2 + rx0
    # save positions as a file
    np.save(pos_filename, pos)
    # save initial conditions for next sample as a file
    init = pos[:,50:60,:] - pos[:,49:59,:]
    np.save(init_filename, init)

save_files()

