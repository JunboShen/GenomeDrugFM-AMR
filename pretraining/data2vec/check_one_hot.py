import os
import numpy as np

# Directories containing the .npy files
data_dir = "./one_hots_data2vec_new"
indices_dir = "./one_hots_split_indices_data2vec_new"

def validate_npy_files(data_dir, indices_dir):
    """
    Validate the .npy files by checking their shapes and lengths.
    """
    data_files = sorted(os.listdir(data_dir))
    indices_files = sorted(os.listdir(indices_dir))
    len_list = []
    for data_file, indices_file in zip(data_files, indices_files):
        try:
            # Load the one-hot encoded data
            data_path = os.path.join(data_dir, data_file)
            data = np.load(data_path)
            
            # Load the indices file
            indices_path = os.path.join(indices_dir, indices_file)
            indices = np.load(indices_path)
            len_list.append(data.shape[0])
            if data.shape[0] < 100:
                # Print the shape of the data and the indices
                print(f"File: {data_file}")
                print(f"  Data shape: {data.shape}")
                print(f"  Len Indices: {len(indices)}")
            
        
        except Exception as e:
            print(f"Error processing {data_file}: {str(e)}")
    print(f"Max length of data files: {max(len_list)}")
    print(f"Min length of data files: {min(len_list)}")
    print(f"Average length of data files: {sum(len_list)/len(len_list)}")
    print(f"Total number of data files: {len(len_list)}")

if __name__ == "__main__":
    validate_npy_files(data_dir, indices_dir)