import re
import argparse
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot training loss over epochs.')
parser.add_argument('--filepath', type=str, required=True, help='Path to the log file')
parser.add_argument('--output', type=str, required=True, help='Path to save the output PNG file')
parser.add_argument('--title', type=str, default='Training Loss Over Epochs', help='Title of the plot')
args = parser.parse_args()

filepath = args.filepath
output_path = args.output

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
plt.title(args.title)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(output_path)
