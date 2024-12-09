import torch
import gym
import torch.optim as optim
import torch.nn as nn
import ale_py
import numpy as np


model = torch.load('checkpoints/spaceinvaders/model.pt')

model.fc = nn.Linear(in_features = model.fc.in_features, out_features=num_pong_actions)
# env = gym.make('Pong-v0')
env = gym.make('SpaceInvadersDeterministic-v4')
# Set random seeds
np.random.seed(args.seed)
env.action_space.seed(args.seed)
env.reset(seed=args.seed)


optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10  # Adjust based on performance

for epoch in range(num_epochs):
    state = env.reset()
    done = False
    while not done:
        # Preprocess state as required
        action = model(state)
        next_state, reward, done, _ = env.step(action)
        loss = criterion(action, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state

torch.save(model, 'checkpoints/spaceinvaders/model_tuned_pong.pt')