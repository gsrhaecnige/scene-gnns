import re
import matplotlib.pyplot as plt

filepath = "checkpoints_transfer/pong_transfer/log.txt"

# Load the log text from a file
with open(filepath, 'r') as file:
	log_text = file.read()

# Use a regex to find lines with average loss:
# The pattern is something like: "====> Epoch: X Average loss: YYYYYY"
epoch_losses = re.findall(r'====> Epoch:\s+(\d+)\s+Average loss:\s+([\d\.]+)', log_text)

epochs = [int(e[0]) for e in epoch_losses]
losses = [float(e[1]) for e in epoch_losses]

# Plot the loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()
