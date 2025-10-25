import numpy as np
import pandas as pd
import os

def load_arr_from_npz(npz_path):
    npz_file_obj = np.load(npz_path)
    arr = npz_file_obj.f.arr_0.copy() # Rely on default name from np.savez
    npz_file_obj.close()
    return arr

if __name__ == '__main__':
    data_dir = 'data'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    # Load BERT embeddings as 2D numpy array
    # Each row corresponds to row of data frames above
    # Each col is one of the H=768 dimensions of BERT embedding    
    xBERT_train_NH = load_arr_from_npz(os.path.join(
        data_dir, 'x_train_BERT_embeddings.npz'))
    assert xBERT_train_NH.ndim == 2
    xBERT_test = load_arr_from_npz(os.path.join(
        data_dir, 'x_test_BERT_embeddings.npz'))
    assert xBERT_train_NH.ndim == 2
