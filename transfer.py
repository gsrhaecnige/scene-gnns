# command
# python transfer.py \
#     --pretrained-model checkpoints/spaceinvaders/model.pt \
#     --new-dataset data/balls_train_50.h5 \
#     --batch-size 512 \
#     --epochs 50 \
#     --learning-rate 1e-4 \
#     --encoder medium \
#     --action-dim 6 --name pong_transfer \
#     --decoder \
#     --device-id 0


import argparse
import torch
import utils
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import modules

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer Learning for SWM Models')
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
    
    # Model parameters
    parser.add_argument('--encoder', type=str, default='medium', help='Object extractor CNN size (e.g., `small`).')
    parser.add_argument('--sigma', type=float, default=0.5, help='Energy scale.')
    parser.add_argument('--hinge', type=float, default=1., help='Hinge threshold parameter.')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Number of hidden units in transition MLP.')
    parser.add_argument('--embedding-dim', type=int, default=4, help='Dimensionality of embedding.')
    parser.add_argument('--action-dim', type=int, default=6, help='Dimensionality of action space.')
    parser.add_argument('--num-objects', type=int, default=3, help='Number of object slots in model.')
    parser.add_argument('--ignore-action', action='store_true', default=False, help='Ignore action in GNN transition model.')
    parser.add_argument('--copy-action', action='store_true', default=True, help='Apply same action to all object slots.')
    
    # Decoder parameters
    parser.add_argument('--decoder', action='store_true', default=False, help='Use decoder and pixel-based loss.')
    
    # Device parameters
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')
    parser.add_argument('--device-id', type=int, default=0, help='CUDA device ID.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--log-interval', type=int, default=20, help='Batches to wait before logging training status.')
    
    # Dataset parameters
    parser.add_argument('--pretrained-model', type=str, required=True, help='Path to the pre-trained model file.')
    parser.add_argument('--new-dataset', type=str, required=True, help='Path to the new dataset (e.g., Pong).')
    
    # Experiment parameters
    parser.add_argument('--name', type=str, default='transfer_experiment', help='Experiment name.')
    parser.add_argument('--save-folder', type=str, default='checkpoints_transfer', help='Path to save checkpoints.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Set up experiment name and directories
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    
    exp_name = args.name if args.name != 'transfer_experiment' else timestamp
    save_folder = os.path.join(args.save_folder, exp_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(save_folder, 'log.txt')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file, 'a'))
    print = logger.info
    
    # Save metadata
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    pickle.dump({'args': args}, open(meta_file, "wb"))
    
    # Set device
    device = torch.device(f'cuda:{args.device_id}' if args.cuda else 'cpu')
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Load new dataset
    print('Loading new dataset...')
    dataset = utils.StateTransitionsDataset(hdf5_file=args.new_dataset)
    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Get input shape from data
    obs = next(iter(train_loader))[0]
    input_shape = obs[0].size()
    
    # Initialize model
    print('Initializing model...')
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
        encoder=args.encoder
    ).to(device)
    # model = SpaceToPong(num_actions=args.action_dim).to(device)


    # Initialize weights (optional, since we are loading pre-trained weights)
    model.apply(utils.weights_init)
    
    # Load pre-trained weights
    print(f'Loading pre-trained model from {args.pretrained_model}...')
    pretrained_state = torch.load(args.pretrained_model, map_location=device)
    filtered_state = {k: v for k, v in pretrained_state.items() if 'some_layer' not in k}
    model.load_state_dict(filtered_state, strict=False)
    # model.load_state_dict(pretrained_state, strict=False)
    
    # Optionally freeze encoder layers to retain learned features
    freeze_encoder = True
    if freeze_encoder:
        print('Freezing encoder layers...')
        for name, param in model.obj_extractor.named_parameters():
            param.requires_grad = False
        for name, param in model.obj_encoder.named_parameters():
            param.requires_grad = False
    
    # Initialize optimizer (only parameters that require gradients)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # Initialize decoder if needed
    if args.decoder:
        print('Initializing decoder...')
        if args.encoder == 'large':
            decoder = modules.DecoderCNNLarge(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape
            ).to(device)
        elif args.encoder == 'medium':
            decoder = modules.DecoderCNNMedium(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape
            ).to(device)
        elif args.encoder == 'small':
            decoder = modules.DecoderCNNSmall(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape
            ).to(device)
        else:
            raise ValueError(f"Unknown encoder type: {args.encoder}")
        
        decoder.apply(utils.weights_init)
        
        # Load decoder weights if available (optional)
        decoder_path = os.path.splitext(args.pretrained_model)[0] + '_decoder.pt'
        if os.path.exists(decoder_path):
            print(f'Loading pre-trained decoder from {decoder_path}...')
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        
        # Initialize decoder optimizer
        optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    
    # Set up saving paths
    model_file = os.path.join(save_folder, 'fine_tuned_model.pt')
    if args.decoder:
        decoder_file = os.path.join(save_folder, 'fine_tuned_decoder.pt')
    
    # Training loop
    print('Starting transfer learning...')
    step = 0
    best_loss = 1e9
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.decoder:
            decoder.train()
        train_loss = 0
        
        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()
            if args.decoder:
                optimizer_dec.zero_grad()
                obs, action, next_obs = data_batch
                objs = model.obj_extractor(obs)
                # print(f"Shape of objs: {objs.shape}")
                # print(f"Shape of obs: {obs.shape}")
                # print(f"Shape of action: {action.shape}")
                # Take a random 100 features of the objs vector
                # objs = objs.view(objs.size(0), -1)  
                # indices = torch.randperm(objs.size(1))[:100]  # Get random 100 indices
                # objs = objs[:, indices]  # Select the random 100 features
                state = model.obj_encoder(objs)

                print(f"Shape of state: {state.shape}")
                
                rec = torch.sigmoid(decoder(state))
                loss = F.binary_cross_entropy(rec, obs, reduction='sum') / obs.size(0)

                next_state_pred = state + model.transition_model(state, action)
                next_rec = torch.sigmoid(decoder(next_state_pred))
                next_loss = F.binary_cross_entropy(next_rec, next_obs, reduction='sum') / obs.size(0)
                loss += next_loss
            else:
                obs, action, next_obs = data_batch
                loss = model.contrastive_loss(obs, action, next_obs)
			
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if args.decoder:
                optimizer_dec.step()
			
            if batch_idx % args.log_interval == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data_batch[0])}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data_batch[0]):.6f}')
			
            step += 1
        
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}')
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)
            if args.decoder:
                torch.save(decoder.state_dict(), decoder_file)
            print(f'Best model saved with loss {best_loss:.6f}')
        
        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_file = os.path.join(save_folder, f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_file)
            if args.decoder:
                checkpoint_decoder_file = os.path.join(save_folder, f'decoder_epoch_{epoch}.pt')
                torch.save(decoder.state_dict(), checkpoint_decoder_file)
            print(f'Saved model checkpoint at epoch {epoch}')
    
    print('Transfer learning completed.')

if __name__ == '__main__':
    main()
