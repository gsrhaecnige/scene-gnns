import torch
import gym
import torch.optim as optim
import torch.nn as nn
import ale_py
import numpy as np
import gymnasium as gym
from gymnasium import logger
import atari_py

# Define your model architecture (example: a custom model class)
class CustomModel(nn.Module):
    def __init__(self, num_actions):
        super(CustomModel, self).__init__()
        self.obj_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.obj_encoder = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
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
        x = x.view(x.size(0), -1)
        x = self.obj_encoder(x)
        return x

model = torch.load('checkpoints/spaceinvaders/model.pt', map_location=torch.device('cpu'))

# Assuming the model's final layer is named 'fc'
num_pong_actions = 6  # Pong typically has 6 discrete actions
model = CustomModel(num_pong_actions)


# Print the model architecture
# print("Model architecture:")
# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}")

model.obj_encoder[-1] = nn.Linear(in_features=model.obj_encoder[-1].in_features, out_features=num_pong_actions)


args_seed = 42 # rtemp we can do argparse later 
np.random.seed(args_seed)

env = gym.make('SpaceInvadersDeterministic-v4')
env.action_space.seed(args_seed)
env.reset(seed=args_seed)

# Initialize the replay buffer
replay_buffer = []

# # Example loop to interact with the environment
# for i in range(100):  # Adjust the number of iterations as needed
#     ob = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()  # Replace with your model's action
#         ob, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#         replay_buffer[-1]['action'].append(action)
#         replay_buffer[-1]['next_obs'].append(ob[1])

#         if done:
#             break

#     if i % 10 == 0:
#         print("iter " + str(i))

# env.close()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10  # Adjust based on performance

for epoch in range(num_epochs):
    state = env.reset()
    state = np.array(state)  # Convert state to numpy array
    state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Preprocess state as required
    done = False
    while not done:
        # Get action from the model
        action_logits = model(state)
        action = torch.argmax(action_logits, dim=1).item()

        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Preprocess next_state
        next_state = np.array(next_state)  # Convert next_state to numpy array
        next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Calculate loss
        reward = torch.tensor([reward], dtype=torch.float32)
        loss = criterion(action_logits, reward)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

torch.save(model, 'checkpoints/spaceinvaders/model_tuned_pong.pt')