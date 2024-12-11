import argparse
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
from collections import defaultdict

import modules

torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of prediction steps to evaluate.')
    parser.add_argument('--dataset', type=str,
                        default='data/shapes_eval.h5',
                        help='Dataset string.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    return parser.parse_args()

def main():
    args_eval = parse_args()

    meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
    model_file = os.path.join(args_eval.save_folder, 'model.pt')

    # Load arguments from metadata
    args = pickle.load(open(meta_file, 'rb'))['args']

    # Override certain arguments with evaluation-specific ones
    args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
    args.batch_size = 100
    args.dataset = args_eval.dataset
    args.seed = 0

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Set device
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Initialize dataset and dataloader
    dataset = utils.PathDataset(
        hdf5_file=args.dataset, path_length=args_eval.num_steps)
    eval_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Optional: Print the length of each batch (can be removed if not needed)
    for batch in eval_loader:
        print(len(batch))
        break  # Remove or adjust as needed

    # Get a sample to determine input shape
    obs = next(iter(eval_loader))[0]
    input_shape = obs[0][0].size()

    # Initialize and load the model
    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Evaluation metrics
    topk = [1]
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0

    pred_states = []
    next_states = []

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(eval_loader):
            # Move data to the appropriate device
            data_batch = [[t.to(device) for t in tensor] for tensor in data_batch]
            observations, actions = data_batch

            # Skip batches that don't match the expected batch size
            if observations[0].size(0) != args.batch_size:
                continue

            obs = observations[0]
            next_obs = observations[-1]

            # Encode observations
            state = model.obj_encoder(model.obj_extractor(obs))
            next_state = model.obj_encoder(model.obj_extractor(next_obs))

            # Predict future states
            pred_state = state
            for i in range(args_eval.num_steps):
                pred_trans = model.transition_model(pred_state, actions[i])
                pred_state = pred_state + pred_trans

            pred_states.append(pred_state.cpu())
            next_states.append(next_state.cpu())

        # Concatenate all predicted and actual next states
        pred_state_cat = torch.cat(pred_states, dim=0)
        next_state_cat = torch.cat(next_states, dim=0)

        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        # Compute pairwise distance matrix
        dist_matrix = utils.pairwise_distance_matrix(
            next_state_flat, pred_state_flat)
        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Stable sort using numpy
        dist_np = dist_matrix_augmented.numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)
        indices = torch.from_numpy(indices).long()

        print('Processed {} samples of size {}'.format(
            batch_idx + 1, args.batch_size))

        # Prepare labels for evaluation
        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples += full_size
        print('Size of current topk evaluation batch: {}'.format(
            full_size))

        # Calculate Hits@k
        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        # Calculate Mean Reciprocal Rank (MRR)
        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum += reciprocal_ranks.sum()

    # Display evaluation metrics
    for k in topk:
        print('Hits @ {}: {:.4f}'.format(k, hits_at[k] / float(num_samples)))

    print('MRR: {:.4f}'.format(rr_sum / float(num_samples)))

if __name__ == "__main__":
    import multiprocessing
    # On macOS, it's recommended to set the start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)
    main()
