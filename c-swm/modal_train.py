import modal

app = modal.App("training-modal")
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "h5py"])

@app.function(gpu="A10G")
def train_model(args):
    import torch
    import numpy as np
    import os
    import pickle
    import datetime
    from torch.utils import data
    import torch.nn.functional as F
    import modules
    import utils
    
    print(f"Received args: {args}")
    # Set up environment
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Set up device
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Dataset and DataLoader
    dataset = utils.StateTransitionsDataset(hdf5_file=args.dataset)
    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    obs = next(iter(train_loader))[0]
    input_shape = obs[0].size()
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
    model.apply(utils.weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate)

    # Optional decoder
    if args.decoder:
        decoder = modules.DecoderCNNSmall(
            input_dim=args.embedding_dim,
            num_objects=args.num_objects,
            hidden_dim=args.hidden_dim // 16,
            output_size=input_shape).to(device)
        decoder.apply(utils.weights_init)
        optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

    # Training loop
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            if args.decoder:
                optimizer_dec.zero_grad()
                obs, action, next_obs = data_batch
                objs = model.obj_extractor(obs)
                state = model.obj_encoder(objs)

                rec = torch.sigmoid(decoder(state))
                loss = F.binary_cross_entropy(rec, obs, reduction='sum') / obs.size(0)
                next_state_pred = state + model.transition_model(state, action)
                next_rec = torch.sigmoid(decoder(next_state_pred))
                next_loss = F.binary_cross_entropy(next_rec, next_obs, reduction='sum') / obs.size(0)
                loss += next_loss
            else:
                loss = model.contrastive_loss(*data_batch)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if args.decoder:
                optimizer_dec.step()

        avg_loss = train_loss / len(train_loader.dataset)
        if avg_loss < best_loss:
            best_loss = avg_loss
            with open(args.model_file, "wb") as f:
                torch.save(model.state_dict(), f)

@app.local_entrypoint()
# def main(dataset: str, encoder: str, embedding_dim: int, num_objects: int, ignore_action: bool, name: str):
def main():
    
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=1024,
						help='Batch size.')
	parser.add_argument('--epochs', type=int, default=100,
						help='Number of training epochs.')
	parser.add_argument('--learning-rate', type=float, default=5e-4,
						help='Learning rate.')

	parser.add_argument('--encoder', type=str, default='small',
						help='Object extrator CNN size (e.g., `small`).')
	parser.add_argument('--sigma', type=float, default=0.5,
						help='Energy scale.')
	parser.add_argument('--hinge', type=float, default=1.,
						help='Hinge threshold parameter.')

	parser.add_argument('--hidden-dim', type=int, default=512,
						help='Number of hidden units in transition MLP.')
	parser.add_argument('--embedding-dim', type=int, default=2,
						help='Dimensionality of embedding.')
	parser.add_argument('--action-dim', type=int, default=4,
						help='Dimensionality of action space.')
	parser.add_argument('--num-objects', type=int, default=5,
						help='Number of object slots in model.')
	parser.add_argument('--ignore-action', action='store_true', default=False,
						help='Ignore action in GNN transition model.')
	parser.add_argument('--copy-action', action='store_true', default=False,
						help='Apply same action to all object slots.')

	parser.add_argument('--decoder', action='store_true', default=False,
						help='Train model using decoder and pixel-based loss.')

	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='Disable CUDA training.')
	parser.add_argument('--seed', type=int, default=42,
						help='Random seed (default: 42).')
	parser.add_argument('--log-interval', type=int, default=20,
						help='How many batches to wait before logging'
							'training status.')
	parser.add_argument('--dataset', type=str,
						default='data/shapes_train.h5',
						help='Path to replay buffer.')
	parser.add_argument('--name', type=str, default='none',
						help='Experiment name.')
	parser.add_argument('--save-folder', type=str,
						default='checkpoints',
						help='Path to checkpoints.')
	parser.add_argument("-f", required=False)

	args = parser.parse_args()
  
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.model_file = os.path.join(args.save_folder, f"{args.name}_model.pt")
	os.makedirs(args.save_folder, exist_ok=True)
	train_model.remote({"dataset": "data/balls_train.h5", "encoder": "medium", "name": "balls", "ignore-action": False, })

	train_model.remote(args)
