import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import modal

app = modal.App("example-get-started")
image = modal.Image.debian_slim().pip_install(["torch", "numpy", "h5py", "matplotlib", "argparse", "numpy", "utils","modules", "wandb",  "datetime"])
# Define a simple neural network
class SimpleModel(nn.Module):
	def __init__(self):
		super(SimpleModel, self).__init__()
		self.fc1 = nn.Linear(10, 50)
		self.fc2 = nn.Linear(50, 1)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)
		return x

# Generate some random data
x_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

# Create DataLoader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

@app.function(image=image)
def train_model():
	num_epochs = 20
	for epoch in range(num_epochs):
		for inputs, targets in train_loader:
			# Forward pass
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			# Backward pass and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

	print("Training complete.")

@app.local_entrypoint()
def main():
	train_model.remote()

if __name__ == "__main__":
	main()