import torch
import gym
import torch.optim as optim
import torch.nn as nn
import ale_py
import numpy as np
import gymnasium as gym
from gymnasium import logger
import atari_py
import utils
from torch.utils import data

import pickle
import os
import datetime
import cv2
from collections import OrderedDict

# Load the Pong training dataset
# train_dataset = utils.StateTransitionsDataset(hdf5_file='data/pong_train.h5')
# train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# # Load the Pong evaluation dataset
# eval_dataset = utils.StateTransitionsDataset(hdf5_file='data/pong_eval.h5')
# eval_loader = data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Preprocessing function to reduce input size
def preprocess_frame(frame):
    # Resize RGB frame to 84x84 but keep the 3 channels
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    # frame: (84,84,3)
    frame = frame.transpose((2,0,1)) # (3,84,84)
    return frame
# Define your configuration parameters
config = {
    'epochs': 10,
    'learning_rate': 1e-4,
    'seed': 42,
    # 'model_checkpoint': 'checkpoints/spaceinvaders/model.pt',
	'model_checkpoint': 'checkpoints/pong_k3/model.pt',
    'save_folder': 'checkpoints/space_for_pong_test',
    # 'target_game': 'SpaceInvadersDeterministic-v4',
	'target_game': 'Pong-ramDeterministic-v4',
}

config['save_folder'] = (config['save_folder'] + "_" + str(config['epochs']))

# Ensure the save directory exists
if not os.path.exists(config['save_folder']):
    os.makedirs(config['save_folder'])

# Generate a timestamp for unique identification
timestamp = datetime.datetime.now().isoformat()

# Define the path for the metadata file
meta_file = os.path.join(config['save_folder'], f'metadata.pkl')

# Save the configuration parameters to the metadata file
with open(meta_file, 'wb') as f:
    pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'Configuration parameters saved to {meta_file}')


# Define your model architecture (example: a custom model class)
class PongToSpace(nn.Module):
    def __init__(self, num_actions):
        super(PongToSpace, self).__init__()
        self.obj_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.obj_encoder = nn.Sequential(
            nn.Linear(32 * 210 * 160, 128),  # Adjust the input size based on the output of obj_extractor
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.transition_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        x = self.obj_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"Shape after flattening: {x.shape}")  # Debugging information
        x = self.obj_encoder(x)
        return x


class SpaceToPong(nn.Module):
    def __init__(self, num_actions=4):
        super(SpaceToPong, self).__init__()
        
        # According to your parameters:
        # obj_extractor:
        # - cnn1.weight: [32,6,9,9] suggests Conv2d(in_channels=6, out_channels=32, kernel_size=9)
        # - ln1.* parameters suggest a batch/instance/layer norm with 32 channels
        # - cnn2.weight: [3,32,5,5] suggests Conv2d(in_channels=32, out_channels=3, kernel_size=5)
        self.obj_extractor = nn.Sequential(OrderedDict([
            ('cnn1', nn.Conv2d(in_channels=6, out_channels=32, kernel_size=9)),
            ('ln1', nn.BatchNorm2d(32)),   # chosen BatchNorm2d based on running_mean/var
            ('cnn2', nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5))
        ]))

        # obj_encoder:
        # - fc1.weight: [512,100] means nn.Linear(100->512)
        # - fc2.weight: [512,512] means nn.Linear(512->512)
        # - fc3.weight: [4,512] means nn.Linear(512->4)
        # - ln.*: [512] suggests a normalization layer with 512 features (likely LayerNorm or BatchNorm1d)
        # We'll assume LayerNorm(512)
        self.obj_encoder = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(100, 512)),
            ('fc2', nn.Linear(512, 512)),
            ('ln', nn.LayerNorm(512)),
            ('fc3', nn.Linear(512, 4))
        ]))
        
        # transition_model:
        # edge_mlp and node_mlp have several layers:
        # edge_mlp:
        # - Linear(8->512), Linear(512->512), another Linear(512->512)? The pattern suggests a 3-layer MLP
        # node_mlp:
        # - Linear(522->512), Linear(512->512), Linear(512->4)
        # The presence of weights like edge_mlp.3.weight: [512] suggests additional normalization or bias terms.
        # We'll guess a structure based on the indices:
        # Typically, layers in .0, .2, .5 might be linear layers and .3 something else like a norm or activation.
        # Without exact code, we can try:
        
        self.transition_model = nn.ModuleDict({
            'edge_mlp': nn.Sequential(
                nn.Linear(8, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            ),
            'node_mlp': nn.Sequential(
                nn.Linear(522, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 4)
            )
        })

    def forward(self, x):
        # Forward logic depends on your data flow.
        # For demonstration, we'll just show dimension transformations.
        # After obj_extractor, you must ensure x matches the shape needed for fc1 (which expects 100 inputs).
        # The actual input image shape and pre-processing are crucial here.

        # Example forward (this may not run until you have correct input shapes):
        x = self.obj_extractor(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Suppose after flattening, we have exactly 100 features:
        x = self.obj_encoder(x)  # produces a (batch_size,4) output
        # Using transition_model would require separate inputs (like edges or node attributes)
        # This code is a placeholder.

        return x


np.random.seed(config['seed'])

env = gym.make('SpaceInvadersDeterministic-v4')
num_actions = env.action_space.n
print("Action Space: " + str(num_actions))
env.action_space.seed(config['seed'])
env.reset(seed=config['seed'])


pretrained_model = torch.load(config['model_checkpoint'], map_location='cpu')
model = SpaceToPong(num_actions)
model.load_state_dict(pretrained_model, strict=False)

# Freeze obj_extractor and obj_encoder layers
for param in model.obj_extractor.parameters():
    param.requires_grad = False
for param in model.obj_encoder.parameters():
    param.requires_grad = False
# # Check requires_grad status
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")

# Print the model architecture
print("Model architecture:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# print(type(model))
# print("Model architecture:")
# for name, value in model.items():
#     print(f"{name}: {value.shape}")

# print(model[])

# Model architecture:
	# obj_extractor.0.weight: torch.Size([16, 3, 3, 3])
	# obj_extractor.0.bias: torch.Size([16])
	# obj_extractor.2.weight: torch.Size([32, 16, 3, 3])
	# obj_extractor.2.bias: torch.Size([32])
	# obj_encoder.0.weight: torch.Size([128, 1075200])
	# obj_encoder.0.bias: torch.Size([128])
	# obj_encoder.2.weight: torch.Size([64, 128])
	# obj_encoder.2.bias: torch.Size([64])
	# obj_encoder.4.weight: torch.Size([6, 64])
	# obj_encoder.4.bias: torch.Size([6])
	# transition_model.0.weight: torch.Size([128, 64])
	# transition_model.0.bias: torch.Size([128])
	# transition_model.2.weight: torch.Size([64, 128])
	# transition_model.2.bias: torch.Size([64])
	# transition_model.4.weight: torch.Size([32, 64])
	# transition_model.4.bias: torch.Size([32])

# model.obj_encoder[-1] = nn.Linear(in_features=model.obj_encoder[-1].in_features, out_features=num_actions)
# ^^ this deals with differences in action spaces

# Reinitialize the weights of the new output layer
# nn.init.kaiming_uniform_(model.obj_encoder[2].weight, nonlinearity='relu')
# nn.init.zeros_(model.obj_encoder[2].bias)

learning_rate = 0.0001  # Adjust based on performance
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate']) # only update the ones we want to change
criterion = nn.CrossEntropyLoss()

num_epochs = config['epochs']  # Adjust based on performance

avg_loss = 0
ct = 0
for epoch in range(num_epochs):
	# state = env.reset()
	state, _ = env.reset(seed=config['seed'])
	# if isinstance(state, tuple) or isinstance(state, list):
		# state = np.array(state[0])  # Assuming the first element is the observation
	state = preprocess_frame(state)  # shape: (1,84,84)
	print("Preprocessed State shape:" + str(state.shape))
	state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Preprocess state as required
	# state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Preprocess state as required
	print("Input State shape:" + str(state.shape))
	done = False
	
	print("Epoch: " + str(epoch))
	avg_loss = avg_loss/ct if (ct > 0) else avg_loss/1
	print("Average Loss: " + str(avg_loss))
	avg_loss = 0
	ct = 0
	while not done:

		# Get action from the model
		action_logits = model(state)
		action = torch.argmax(action_logits, dim=1).item()

		# Step the environment
		next_state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated

		# Preprocess next_state
		# print(f"Next state type: {type(next_state)}, shape: {np.array(next_state).shape if isinstance(next_state, np.ndarray) else 'N/A'}")  # Debugging information
		if isinstance(next_state, tuple) or isinstance(next_state, list):
			# next_state = np.array(next_state[0])  # Assuming the first element is the observation
			next_state = preprocess_frame(next_state)
		next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

		# Calculate loss
		action_tensor = torch.tensor([action], dtype=torch.long)
		loss = criterion(action_logits, action_tensor)
		avg_loss += loss
		ct += 1

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		state = next_state

torch.save(model, f"{config['save_folder']}/model_{num_epochs}.pt")

# eval 
# python eval.py --dataset data/pong_eval.h5 --save-folder checkpoints/transfer_learning --num-steps 1