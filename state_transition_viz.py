
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import StateTransitionsDataset
import pickle
import os
import argparse
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

def visualize_object_masks(model: torch.nn.Module, obs: torch.Tensor, save_path: Optional[str] = None) -> None:
    """Visualize object masks extracted by the CNN encoder."""
    # Get object masks
    with torch.no_grad():
        masks = model.obj_extractor(obs)  # [B, num_objects, H, W]
    
    # Plot masks for each object
    batch_size = masks.size(0)
    num_objects = masks.size(1)
    
    for b in range(batch_size):
        fig, axes = plt.subplots(1, num_objects + 1, figsize=(3*(num_objects + 1), 3))
        
        # Plot original image
        if obs.size(1) == 3:  # RGB image
            axes[0].imshow(obs[b].permute(1, 2, 0).cpu())
        else:  # Grayscale image
            axes[0].imshow(obs[b][0].cpu(), cmap='gray')
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        # Plot masks
        for i in range(num_objects):
            axes[i+1].imshow(masks[b, i].cpu(), cmap='viridis')
            axes[i+1].set_title(f'Object {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/masks_batch{b}.png')
        plt.show()
        plt.close()

def visualize_transitions(model: torch.nn.Module, 
                          predicted_next_states,
                          num_batches,
                          save_path: Optional[str] = None) -> None:
    """Visualize state transitions in 2D latent space for a specific object (or all)."""

    # print(predicted_next_states[0])
    # print(predicted_next_states.shape)
    state0 = predicted_next_states[0]
    num_objects = state0.size(1)
    
    for b in range(num_batches):
        all_points = []
        
        for idx in range(predicted_next_states.shape[0]):
            state = predicted_next_states[idx]
            embedding_dim = state.size(2)
            # Get states for this batch
            batch_state = state[b].cpu().numpy()  # [num_objects, embedding_dim]
        
            # If dimension > 2, use PCA to reduce to 2D
            if embedding_dim > 2:
                # Fit PCA on all points to ensure consistent transformation
                batch_state = np.concatenate([batch_state])
                pca = PCA(n_components=2)
                pca.fit(batch_state)
            
                # Transform all points
                batch_state = pca.transform(batch_state)
                all_points.append(batch_state)
            elif embedding_dim == 1:
                # If 1D, add a zero column for y-axis
                batch_state = np.column_stack([batch_state, np.zeros_like(batch_state)])
                batch_next_state = np.column_stack([batch_next_state, np.zeros_like(batch_next_state)])
                batch_pred_next_state = np.column_stack([batch_pred_next_state, np.zeros_like(batch_pred_next_state)])
                batch_pred_trans = np.column_stack([batch_pred_trans, np.zeros_like(batch_pred_trans)])
            
        for object_idx in range(num_objects):
            colors = np.linspace(0.4, 0, len(predicted_next_states)) 
            fig, ax = plt.subplots(figsize=(10, 10))
            x_coords = [all_points[step][object_idx][0] for step in range(50)]
            y_coords = [all_points[step][object_idx][1] for step in range(50)]

            for i in range(len(predicted_next_states)):
                # alpha = (i + 2 * len(predicted_next_states) / 3) / (2 * len(predicted_next_states))  # Alpha increases as i increases
                # ax.scatter(x_coords[i], y_coords[i], color="blue", alpha=alpha, s=100)
                # alpha = (i + 2 * len(predicted_next_states) / 3) / (2 * len(predicted_next_states))  # Alpha increases as i increases
                # Calculate the color for the current point based on its position
                point_color = plt.cm.RdYlGn(colors[i])  # Use the RdYlGn colormap to get a color between green and red
                # Plot the point with the calculated color and alpha
                ax.scatter(x_coords[i], y_coords[i], color=point_color, alpha=1, s=100)
        
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            plt.title(f'State Transitions (Batch {2}, Object {object_idx+1})')

            cmap = plt.cm.YlOrRd  # Yellow to Red colormap (you can use RdYlGn or other colormaps as well)
            norm = mcolors.Normalize(vmin=0, vmax=len(x_coords) - 1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Empty array for the color bar
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_ticks([])
            cbar.set_label('Yellow = Old, Red = New')
            # plt.legend()
            if save_path:
                plt.savefig(f'{save_path}/transitions_batch{2}_object{object_idx}.png')
            plt.show()
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/pong_k5/model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--meta-path', type=str, default='checkpoints/pong_k5/metadata.pkl',
                       help='Path to metadata file')
    parser.add_argument('--dataset', type=str, default='data/pong_eval.h5',
                       help='Path to dataset')
    parser.add_argument('--save-dir', type=str, default='pong_k5_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of examples to visualize')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='Disable CUDA')
    parser.add_argument('--forward', type=str, default="all",
                       help='Either "all" or "one" — how many steps to predict')
    args = parser.parse_args()

    # Load model and metadata
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Load metadata first to get model parameters
    meta_data = pickle.load(open(args.meta_path, 'rb'))
    model_args = meta_data['args']
    
    # Load dataset
    dataset = StateTransitionsDataset(args.dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True)
    
    # Get a batch of data
    obs, action, next_obs = next(iter(data_loader))
    obs = obs.to(device)
    action = action.to(device)
    next_obs = next_obs.to(device)

    # Load model
    import modules
    model = modules.ContrastiveSWM(
        embedding_dim=model_args.embedding_dim,
        hidden_dim=model_args.hidden_dim,
        action_dim=model_args.action_dim,
        input_dims=obs[0].size(),
        num_objects=model_args.num_objects,
        sigma=model_args.sigma,
        hinge=model_args.hinge,
        ignore_action=model_args.ignore_action,
        copy_action=model_args.copy_action,
        encoder=model_args.encoder).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

     # Initialize an empty list to store the predicted states
    predicted_states = []

    if args.forward:
        masks = model.obj_extractor(obs)
        state = model.obj_encoder(masks)

        with torch.no_grad():
            for i in range(50):
                # Get predicted next state
                pred_trans = model.transition_model(state, action)
                pred_next_state = state + pred_trans
                predicted_states.append(pred_next_state)
                state = pred_next_state
    else:
        with torch.no_grad():
            for i in range(50):
                masks = model.obj_extractor(obs)
                state = model.obj_encoder(masks)
                
                # Get predicted next state
                pred_trans = model.transition_model(state, action)
                pred_next_state = state + pred_trans
                predicted_states.append(pred_next_state)
                
                obs, action, next_obs = next(iter(data_loader))
                obs = obs.to(device)

    # Convert predicted states list into a tensor if needed
    predicted_states_tensor = torch.stack(predicted_states)

    # Build and visualize the state transition graph per object slot for the next 50 states
    print("Building state transition graph...")
    num_batches = args.batch_size
    visualize_transitions(model, predicted_states_tensor, num_batches, args.save_dir)
    # visualize_object_masks(model, obs, args.save_dir)

if __name__ == "__main__":
    main()