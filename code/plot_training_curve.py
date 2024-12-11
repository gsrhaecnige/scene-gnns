import matplotlib.pyplot as plt
import re
import numpy as np
import argparse

def extract_average_losses(log_file):
    epochs = []
    losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if '====> Epoch:' in line:
                # Extract epoch number and average loss
                match = re.search(r'====> Epoch: (\d+) Average loss: ([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    epochs.append(epoch)
                    losses.append(loss)
    
    return epochs, losses

def plot_training_curve(epochs, losses, title, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Training Loss Curve - {title}')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss curve from log file')
    parser.add_argument('--log_file', type=str, default='checkpoints/shapes/log.txt',
                        help='Path to the training log file')
    parser.add_argument('--title', type=str, default='Training Progress',
                        help='Title for the plot')
    parser.add_argument('--output', type=str, default='training_curve.png',
                        help='Output file name for the plot')
    
    args = parser.parse_args()
    
    # Extract data
    epochs, losses = extract_average_losses(args.log_file)
    
    # Plot and save
    plot_training_curve(epochs, losses, args.title, args.output)
    print(f"Training curve has been saved as '{args.output}'")
    
    # Print some statistics
    print(f"\nTraining Statistics:")
    print(f"Number of epochs: {len(epochs)}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f} (Epoch {epochs[np.argmin(losses)]})")
