import h5py

# Path to your .h5 file
h5_file_path = '/home/yc774/ondemand/OOD/cellinfinity/Paired-Library/DeepCas9/DeepCas9_Final/model.weights.h5'

# Open the .h5 file and list all groups (layers) and datasets (weights)
with h5py.File(h5_file_path, 'r') as h5_file:
    print("Inspecting .h5 file contents...")

    # Function to recursively print the structure of the file
    def print_structure(item, indent=0):
        whitespace = ' ' * indent
        if isinstance(item, h5py.File):
            print(whitespace + item.filename)
        elif isinstance(item, h5py.Group):
            print(whitespace + item.name)
        elif isinstance(item, h5py.Dataset):
            print(whitespace + item.name + ', shape: ' + str(item.shape))
        else:
            return

        if isinstance(item, h5py.Group) or isinstance(item, h5py.File):
            for sub_item in item.values():
                print_structure(sub_item, indent + 4)

    # Start the recursive printing
    print_structure(h5_file)

