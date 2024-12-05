import h5py
import matplotlib.pyplot as plt
import numpy as np


def visualize_pong_images(n_images=10, data_path='data/pong_train.h5', frame_stack_idx=0, consecutive=False, start_idx=0):
    """
    Visualize a sample of images from the Pong dataset.
    
    Args:
        n_images (int): Number of images to display
        data_path (str): Path to the HDF5 file containing the Pong data
        frame_stack_idx (int): Which frame to show when multiple frames are stacked (0-3)
        consecutive (bool): If True, show consecutive frames from the episode instead of random frames
    """
    with h5py.File(data_path, 'r') as f:
        # Get a random episode
        episode_idx = np.random.randint(0, len(f.keys()))
        episode = f[str(episode_idx)]
        
        # Get observations from the episode
        obs = episode['obs'][:]  # Shape should be (T, C, H, W)
        
        # Calculate grid dimensions
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
        
        # Create a figure with subplots in a grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        # Select frames - either consecutive or random
        if consecutive:
            if len(obs) >= n_images:
                # Random starting point that ensures we have enough consecutive frames
                # start_idx = np.random.randint(0, len(obs) - n_images + 1)
                indices = range(start_idx, start_idx + n_images)
            else:
                indices = range(len(obs))
        else:
            if len(obs) > n_images:
                indices = np.random.choice(len(obs), n_images, replace=False)
                indices.sort()  # Show them in temporal order
            else:
                indices = range(len(obs))
            
        # Plot images
        for idx, frame_idx in enumerate(indices):
            row = idx // n_cols
            col = idx % n_cols
            
            # Get the image and handle the stacked frames
            img = obs[frame_idx]
            n_channels = img.shape[0] // 2  # Assuming 2 frames are stacked
            frame = img[frame_stack_idx*n_channels:(frame_stack_idx+1)*n_channels]
            
            # Transpose to (H, W, C)
            img = frame.transpose(1, 2, 0)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Frame {frame_idx}')
        
        # Turn off any unused subplots
        for idx in range(len(indices), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
            
        plt.suptitle(f'Episode {episode_idx}')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Example: Show 12 consecutive frames in a 3x4 grid
    visualize_pong_images(n_images=10, consecutive=True, start_idx=95)
