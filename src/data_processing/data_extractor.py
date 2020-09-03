import pandas as pd
import h5py

class DataExtractor:

    def __init__(self, data_path, data_folder, use_cols, language, model_type, hdf5_data_folder=None):
        self.data_path = data_path
        self.data_folder = data_folder
        self.use_cols = use_cols
        self.language = language
        self.model_type = model_type
        self.hdf5_data_folder = hdf5_data_folder if hdf5_data_folder else 'preprocessed_data'

    def read_data(self, scope, n_splits):
        def get_file(scope, split_ind):
            return pd.read_json(f"{self.data_path}/{self.data_folder}/{self.language}_{scope}_{split_ind}.jsonl", lines=True)[self.use_cols]

        train_df = get_file(scope, split_ind=0)
        for i in range(1, n_splits):
            train_df = train_df.append(get_file(scope, split_ind=i), ignore_index=True)
        return train_df

    def write_hdf5_data(self, dataset, dataset_name):
        with h5py.File(f'{self.data_path}/{self.hdf5_data_folder}/{self.model_type}_encoder/{dataset_name}.h5', 'w') as hf:
            hf.create_dataset(dataset_name, data=dataset)

    def read_hdf5_data(self, dataset_name, start_index=0, end_index=-1):
        with h5py.File(f'{self.data_path}/{self.hdf5_data_folder}/{self.model_type}_encoder/{dataset_name}.h5', "r") as f:
            dataset = f[dataset_name]
            end_index = end_index if end_index > 0 else dataset.size
            res = dataset[start_index:end_index]
        return res