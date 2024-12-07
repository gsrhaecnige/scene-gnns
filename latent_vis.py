import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import StateTransitionsDataset
import pickle
import os
import argparse
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA

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

def visualize_embeddings(model: torch.nn.Module, obs: torch.Tensor, save_path: Optional[str] = None) -> None:
    """Visualize object embeddings in 2D space, using PCA if needed."""
    with torch.no_grad():
        masks = model.obj_extractor(obs)
        embeddings = model.obj_encoder(masks)  # [B, num_objects, embedding_dim]
    
    batch_size = embeddings.size(0)
    num_objects = embeddings.size(1)
    embedding_dim = embeddings.size(2)
    
    # Note: plotting colors do not necessarily match the actual colors of the objects
    colors = plt.cm.rainbow(np.linspace(0, 1, num_objects))
    
    for b in range(batch_size):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get embeddings for this batch
        batch_embeddings = embeddings[b].cpu().numpy()  # [num_objects, embedding_dim]
        
        # If dimension > 2, use PCA to reduce to 2D
        if embedding_dim > 2:
            pca = PCA(n_components=2)
            batch_embeddings = pca.fit_transform(batch_embeddings)
        elif embedding_dim == 1:
            # If 1D, add a zero column for y-axis
            batch_embeddings = np.column_stack([batch_embeddings, np.zeros_like(batch_embeddings)])
        
        # Plot embeddings
        for i in range(num_objects):
            ax.scatter(batch_embeddings[i, 0],
                      batch_embeddings[i, 1],
                      c=[colors[i]], label=f'Object {i+1}')
        
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        plt.title(f'Object Embeddings (Batch {b})')
        plt.legend()
        if save_path:
            plt.savefig(f'{save_path}/embeddings_batch{b}.png')
        plt.show()
        plt.close()

def visualize_transitions(model: torch.nn.Module, 
                        obs: torch.Tensor, 
                        action: torch.Tensor,
                        next_obs: torch.Tensor,
                        save_path: Optional[str] = None) -> None:
    """Visualize state transitions in 2D latent space, using PCA if needed."""
    with torch.no_grad():
        # Get current and next state embeddings
        masks = model.obj_extractor(obs)
        next_masks = model.obj_extractor(next_obs)
        
        state = model.obj_encoder(masks)
        next_state = model.obj_encoder(next_masks)
        
        # Get predicted next state
        pred_trans = model.transition_model(state, action)
        pred_next_state = state + pred_trans
    
    batch_size = state.size(0)
    num_objects = state.size(1)
    embedding_dim = state.size(2)
    
    # Note: plotting colors do not necessarily match the actual colors of the objects
    colors = plt.cm.rainbow(np.linspace(0, 1, num_objects))
    
    for b in range(batch_size):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get states for this batch
        batch_state = state[b].cpu().numpy()  # [num_objects, embedding_dim]
        batch_next_state = next_state[b].cpu().numpy()
        batch_pred_next_state = pred_next_state[b].cpu().numpy()
        batch_pred_trans = pred_trans[b].cpu().numpy()
        
        # If dimension > 2, use PCA to reduce to 2D
        if embedding_dim > 2:
            # Fit PCA on all points to ensure consistent transformation
            all_points = np.concatenate([batch_state, batch_next_state, batch_pred_next_state])
            pca = PCA(n_components=2)
            pca.fit(all_points)
            
            # Transform all points
            batch_state = pca.transform(batch_state)
            batch_next_state = pca.transform(batch_next_state)
            batch_pred_next_state = pca.transform(batch_pred_next_state)
            # Transform transitions (these are vectors, not points)
            batch_pred_trans = pca.transform(batch_pred_trans)
        elif embedding_dim == 1:
            # If 1D, add a zero column for y-axis
            batch_state = np.column_stack([batch_state, np.zeros_like(batch_state)])
            batch_next_state = np.column_stack([batch_next_state, np.zeros_like(batch_next_state)])
            batch_pred_next_state = np.column_stack([batch_pred_next_state, np.zeros_like(batch_pred_next_state)])
            batch_pred_trans = np.column_stack([batch_pred_trans, np.zeros_like(batch_pred_trans)])
        
        for i in range(num_objects):
            # Plot current state
            ax.scatter(batch_state[i, 0],
                      batch_state[i, 1],
                      c=[colors[i]], label=f'Object {i+1} (Current)')
            
            # Plot predicted next state
            ax.scatter(batch_pred_next_state[i, 0],
                      batch_pred_next_state[i, 1],
                      c=[colors[i]], marker='x', label=f'Object {i+1} (Predicted)')
            
            # Plot actual next state
            ax.scatter(batch_next_state[i, 0],
                      batch_next_state[i, 1],
                      c=[colors[i]], marker='+', label=f'Object {i+1} (Actual)')
            
            # Draw arrows for transitions
            ax.quiver(batch_state[i, 0],
                     batch_state[i, 1],
                     batch_pred_trans[i, 0],
                     batch_pred_trans[i, 1],
                     color=colors[i], alpha=0.5)
        
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        plt.title(f'State Transitions (Batch {b})')
        plt.legend()
        if save_path:
            plt.savefig(f'{save_path}/transitions_batch{b}.png')
        plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/pong/model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--meta-path', type=str, default='checkpoints/pong/metadata.pkl',
                       help='Path to metadata file')
    parser.add_argument('--dataset', type=str, default='data/pong_eval.h5',
                       help='Path to dataset')
    parser.add_argument('--save-dir', type=str, default='pong_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of examples to visualize')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='Disable CUDA')
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Load model and metadata
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Load metadata first to get model parameters
    meta_data = pickle.load(open(args.meta_path, 'rb'))
    model_args = meta_data['args']
    
    # Load dataset
    dataset = StateTransitionsDataset(args.dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)
    
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

    # Visualize different components
    print("Visualizing object masks...")
    visualize_object_masks(model, obs, args.save_dir)
    
    print("Visualizing object embeddings...")
    visualize_embeddings(model, obs, args.save_dir)
    
    print("Visualizing state transitions...")
    visualize_transitions(model, obs, action, next_obs, args.save_dir)

if __name__ == '__main__':
    main()