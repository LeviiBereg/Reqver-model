import pandas as pd


class DataExtractor:
    '''
    Handler of reading and writing procedures for method-description pairs 
        stored in prepared json lines

    Attributes:
        data_path (string): path to folder containing the dataset
        use_cols (list): list of column names to read
        language (string): programming language of extracted pairs
        model_type (string): name of model used for source code encoding
        hdf5_data_folder (string): name of folder containing prepared hdf5 data files
    '''

    def __init__(self, data_path, data_folder, use_cols, language, model_type, hdf5_data_folder=None):
        self.data_path = data_path
        self.data_folder = data_folder
        self.use_cols = use_cols
        self.language = language
        self.model_type = model_type
        self.hdf5_data_folder = hdf5_data_folder if hdf5_data_folder else 'preprocessed_data'

    def read_data(self, scope, n_splits):
        '''
        Read the method-description pairs of `scope` from `n_splits` json lines files

        Args:
            scope (string): scope of retrieved pairs from train, valid or test
            n_splits (list): number of dataset splits to read
        Returns:
            train_df (dataframe): retrieved text pairs
        '''
        def get_file(scope, split_ind):
            return pd.read_json(f"{self.data_path}/{self.data_folder}/{self.language}_{scope}_{split_ind}.jsonl", lines=True)[self.use_cols]

        train_df = get_file(scope, split_ind=0)
        for i in range(1, n_splits):
            train_df = train_df.append(get_file(scope, split_ind=i), ignore_index=True)
        return train_df

    def write_hdf5_data(self, dataset, dataset_name):
        '''
        Writes the `dataset` to `dataset_name` folder

        Args:
            dataset (dataframe): the dataset to store
            dataset_name (string): name of folder and dataset
        Returns:
            -
        '''
        with h5py.File(f'{self.data_path}/{self.hdf5_data_folder}/{self.model_type}_encoder/{dataset_name}.h5', 'w') as hf:
            hf.create_dataset(dataset_name, data=dataset)

    def read_hdf5_data(self, dataset_name, start_index=0, end_index=-1):
        '''
        Reads the pairs from `dataset_name` starting at `start_index` until `end_index`

        Args:
            dataset_name (string): name of folder and dataset
            start_index (int): the start index to read from
            end_index (int): the last index to read up to
        Returns:
            Dataframe of retrieved pairs
        '''
        with h5py.File(f'{self.data_path}/{self.hdf5_data_folder}/{self.model_type}_encoder/{dataset_name}.h5', "r") as f:
            dataset = f[dataset_name]
            end_index = end_index if end_index > 0 else dataset.size
            res = dataset[start_index:end_index]
        return res