import h5py

# Open the h5 file in read mode
with h5py.File('data/balls_train.h5', 'r') as file:
    # Function to recursively list all groups and datasets
    def print_h5_contents(obj, level=0):
        if isinstance(obj, h5py.Group):
            print("  " * level + f"Group: {obj.name}")
            for key in obj.keys():
                print_h5_contents(obj[key], level + 1)
        elif isinstance(obj, h5py.Dataset):
            print("  " * level + f"Dataset: {obj.name}")

    # Start by printing the top level groups and datasets
    print("Groups and Datasets in the file:")
    for key in file.keys():
        print_h5_contents(file[key])
