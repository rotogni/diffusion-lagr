import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def create_frustum_mask(points, fov_angle=45, initial_radius=0.1, near=-1, far=1.0):
    """
    Create a frustum mask for an array of 3D points with Y as the depth axis.
    Exclusively works with PyTorch tensors.
    
    Args:
        points: Tensor of shape (N, K, 3) or (3, K) where:
               - N is the number of trajectories
               - K is the number of points per trajectory
               - For (N, K, 3): dim 2 contains [x, y, z] where y is depth
               - For (3, K): rows are [x, y, z] where z is depth
        fov_angle: Field of view angle in degrees
        initial_radius: Initial radius of the frustum at y=-1
        near: Near plane distance (minimum y value to consider, typically -1)
        far: Far plane distance (maximum y value to consider, typically 1)
        
    Returns:
        Binary mask with 1 for points inside frustum, 0 for points outside
    """
    device = points.device
    dtype = points.dtype
    
    # Get the dimensions and reshape if needed
    if len(points.shape) == 2 and points.shape[1] == 3:
        # Handle case where points is just (K, 3)
        points_reshaped = points.unsqueeze(0)  # Add batch dimension
    elif len(points.shape) == 2 and points.shape[0] == 3:
        # Handle case where points is (3, K)
        num_points = points.shape[1]
        points_reshaped = torch.zeros((1, num_points, 3), device=device, dtype=dtype)
        points_reshaped[0, :, 0] = points[0]  # x
        points_reshaped[0, :, 1] = points[2]  # y (depth) from z
        points_reshaped[0, :, 2] = points[1]  # z from y
    else:
        # Already in (N, K, 3) format
        points_reshaped = points
    
    N, K, _ = points_reshaped.shape
    
    # Initialize mask tensor
    mask = torch.zeros((N, K), device=device, dtype=dtype)
    
    # Convert FOV to radians
    fov_rad = math.radians(fov_angle)
    
    # Vectorized implementation for efficiency
    # Extract coordinates
    x = points_reshaped[:, :, 0]  # (N, K)
    y = points_reshaped[:, :, 1]  # (N, K) depth
    z = points_reshaped[:, :, 2]  # (N, K)
    
    # Create depth mask (points within depth range)
    depth_mask = (y >= near) & (y <= far)
    
    # Normalize depth from [-1, 1] to [0, 1] for radius calculation
    y_normalized = (y - near) / (far - near)
    
    # Calculate maximum allowed radius at each depth
    # Linear interpolation from initial_radius to max_radius
    max_radius = initial_radius + (y_normalized * (math.tan(fov_rad) - initial_radius))
    
    # Calculate distance from central axis (using x and z)
    distance = torch.sqrt(x**2 + z**2)
    
    # Check if points are inside frustum (within depth range and radius)
    radius_mask = distance <= max_radius
    
    # Combine depth and radius masks
    mask = (depth_mask & radius_mask).float()
    
    # Return in the original format
    if len(points.shape) == 2 and points.shape[1] == 3:
        return mask[0]  # Return (K,) mask
    elif len(points.shape) == 2 and points.shape[0] == 3:
        return mask[0]  # Return (K,) mask
    else:
        return mask  # Return (N, K) mask

def apply_frustum_mask(points, mask):
    """
    Apply a binary mask to points, zeroing out points outside the frustum.
    Exclusively works with PyTorch tensors.
    
    Args:
        points: Tensor of shape (N, K, 3) or (3, K) or (K, 3)
        mask: Binary mask of shape (N, K) or (K,)
        
    Returns:
        Masked points with same shape as input
    """
    device = points.device
    dtype = points.dtype
    
    # Handle different input shapes
    original_shape = points.shape
    
    if len(points.shape) == 2 and points.shape[1] == 3:
        # Case: (K, 3)
        points_reshaped = points.unsqueeze(0)  # (1, K, 3)
        mask_reshaped = mask.unsqueeze(0) if len(mask.shape) == 1 else mask  # (1, K)
    elif len(points.shape) == 2 and points.shape[0] == 3:
        # Case: (3, K)
        K = points.shape[1]
        points_reshaped = torch.zeros((1, K, 3), device=device, dtype=dtype)
        points_reshaped[0, :, 0] = points[0]  # x
        points_reshaped[0, :, 1] = points[2]  # y (depth) from z
        points_reshaped[0, :, 2] = points[1]  # z from y
        mask_reshaped = mask.unsqueeze(0) if len(mask.shape) == 1 else mask  # (1, K)
    else:
        # Already in (N, K, 3) format
        points_reshaped = points
        mask_reshaped = mask
    
    # Apply mask (vectorized)
    # Expand mask to apply to all 3 coordinates: (N, K, 1)
    expanded_mask = mask_reshaped.unsqueeze(-1)
    
    # Multiply by the mask
    masked_points = points_reshaped * expanded_mask
    
    # Return in the original format
    if len(original_shape) == 2 and original_shape[1] == 3:
        # Return as (K, 3)
        return masked_points[0]
    elif len(original_shape) == 2 and original_shape[0] == 3:
        # Return as (3, K)
        result = torch.zeros(original_shape, device=device, dtype=dtype)
        result[0] = masked_points[0, :, 0]  # x
        result[1] = masked_points[0, :, 2]  # y
        result[2] = masked_points[0, :, 1]  # z (depth)
        return result
    else:
        # Return as (N, K, 3)
        return masked_points

def visualize_frustum_and_points(original_points, masked_points, fov_angle=45, initial_radius=0.1):
    """
    Visualize original and masked points in 3D with frustum outline.
    Takes PyTorch tensors as input but converts to NumPy for visualization.
    
    Args:
        original_points: Original points tensor
        masked_points: Masked points tensor
        fov_angle: Field of view angle in degrees
        initial_radius: Initial radius of the frustum at y=-1
    """
    # Convert to numpy for visualization
    original_points_np = original_points.detach().cpu().numpy()
    masked_points_np = masked_points.detach().cpu().numpy()
    
    # Handle different input shapes
    if len(original_points_np.shape) == 3:
        # Shape is (N, K, 3) - flatten to (N*K, 3) for visualization
        flat_original = original_points_np.reshape(-1, 3)
        flat_masked = masked_points_np.reshape(-1, 3)
    elif len(original_points_np.shape) == 2 and original_points_np.shape[1] == 3:
        # Shape is (K, 3) - already in the right format
        flat_original = original_points_np
        flat_masked = masked_points_np
    elif len(original_points_np.shape) == 2 and original_points_np.shape[0] == 3:
        # Shape is (3, K) - convert to (K, 3)
        K = original_points_np.shape[1]
        flat_original = np.zeros((K, 3))
        flat_original[:, 0] = original_points_np[0]  # x
        flat_original[:, 1] = original_points_np[2]  # y (depth) from z
        flat_original[:, 2] = original_points_np[1]  # z from y
        
        flat_masked = np.zeros((K, 3))
        flat_masked[:, 0] = masked_points_np[0]  # x
        flat_masked[:, 1] = masked_points_np[2]  # y (depth) from z
        flat_masked[:, 2] = masked_points_np[1]  # z from y
    
    # Extract coordinates
    x_orig, y_orig, z_orig = flat_original[:, 0], flat_original[:, 1], flat_original[:, 2]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(
        x_orig, y_orig, z_orig,
        color='lightgray', alpha=0.5, label='Original Points'
    )
    
    # Plot masked points (non-zero only)
    non_zero = ~np.all(masked_points_np == 0, axis=-1)
    if len(masked_points_np.shape) == 3:
        flat_non_zero = non_zero.reshape(-1)
        flat_masked = masked_points_np.reshape(-1, 3)
    else:
        flat_non_zero = non_zero
    
    if np.any(flat_non_zero):  # Only plot if there are non-zero points
        x_masked = flat_masked[flat_non_zero, 0]
        y_masked = flat_masked[flat_non_zero, 1]
        z_masked = flat_masked[flat_non_zero, 2]
        
        ax.scatter(
            x_masked, y_masked, z_masked,
            color='blue', alpha=0.8, label='Points Inside Frustum'
        )
    
    # Draw frustum outline
    # Convert FOV to radians
    fov_rad = math.radians(fov_angle)
    
    # 1. Draw depth lines (from near to far)
    theta_values = np.linspace(0, 2*np.pi, 16, endpoint=False)
    for theta in theta_values:
        # Calculate points along the line from initial radius to maximum
        depths = np.linspace(-1, 1, 50)
        
        # Calculate x, y, z coordinates for each point along the line
        x_line = []
        y_line = []
        z_line = []
        
        for depth in depths:
            # Normalize depth from [-1, 1] to [0, 1] for radius calculation
            depth_norm = (depth + 1) / 2
            
            # Interpolate radius from initial_radius to max_radius
            current_radius = initial_radius + (depth_norm * (math.tan(fov_rad) - initial_radius))
            
            # Calculate point coordinates
            x_point = current_radius * np.cos(theta)
            z_point = current_radius * np.sin(theta)
            
            x_line.append(x_point)
            y_line.append(depth)  # Use actual depth (-1 to 1)
            z_line.append(z_point)
        
        # Plot the line
        ax.plot(x_line, y_line, z_line, 'r-', linewidth=1)
    
    # 2. Draw circles at different depths
    depth_values = np.linspace(-1, 1, 6)
    theta = np.linspace(0, 2*np.pi, 100)
    
    for depth in depth_values:
        # Normalize depth from [-1, 1] to [0, 1] for radius calculation
        depth_norm = (depth + 1) / 2
        
        # Interpolate radius from initial_radius to max_radius
        current_radius = initial_radius + (depth_norm * (math.tan(fov_rad) - initial_radius))
        
        # Calculate circle coordinates
        x_circle = current_radius * np.cos(theta)
        z_circle = current_radius * np.sin(theta)
        y_circle = depth * np.ones_like(theta)
        
        # Plot the circle
        ax.plot(x_circle, y_circle, z_circle, 'r-', linewidth=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Points with Frustum (FOV: {fov_angle}°, Initial Radius: {initial_radius})')
    ax.legend()
    
    # Set view limits
    ax.set_ylim(-1, 1)  # Depth from -1 to 1
    
    # Set equal aspect ratio for X and Z
    x_range = max(abs(x_orig.min()), abs(x_orig.max()), 
                 abs(z_orig.min()), abs(z_orig.max()))
    ax.set_xlim(-x_range, x_range)
    ax.set_zlim(-x_range, x_range)
    
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.show()

# Example usage with PyTorch tensors
def generate_test_data():
    
    # Test with (N, K, 3) shape
    N, K = 100, 280
    points_NxKx3 = torch.randn((N, K, 3))
    points_NxKx3 = points_NxKx3.clamp(-1, 1)
    
    return points_NxKx3

# Main execution
if __name__ == "__main__":
    # Generate test data
    points_NxKx3 = generate_test_data()
    
    # Test with (N, K, 3) format
    print("\nTesting with (N, K, 3) format:")
    mask_NKx3 = create_frustum_mask(points_NxKx3, fov_angle=45, initial_radius=0.2)
    print(f"Points shape: {points_NxKx3.shape}, Mask shape: {mask_NKx3.shape}")
    masked_points_NKx3 = apply_frustum_mask(points_NxKx3, mask_NKx3)
    print(f"Masked points shape: {masked_points_NKx3.shape}")
    print(f"Points inside frustum: {mask_NKx3.sum().item()} out of {mask_NKx3.numel()} ({mask_NKx3.sum().item()/mask_NKx3.numel()*100:.1f}%)")
    visualize_frustum_and_points(points_NxKx3, masked_points_NKx3, fov_angle=45, initial_radius=0.2)