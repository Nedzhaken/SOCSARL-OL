import pandas as pd
from pandas import DataFrame
import numpy as np
import os
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TrackletsDataset(Dataset):
    """A torch dataset for work with tracklets."""
    MAGNI_DATASET_NAME_TOKEN = 'tracklets_THOR-Magni'

    def __init__(self, folder_name_dataset: str, transformation: Optional[Callable] = None):
        """
        Initializes the dataset.

        Args:
            folder_name_dataset (str): Directory containing all `.csv` tracklet files.
            transformation (Callable, optional): Optional transformation applied to each sample.
        """
        
        self.folder_name_dataset = folder_name_dataset
        self.transformation = transformation
        self.dataframes_dataset = []
        self.dataframe_all_trajectories = self.load_tracklets_from_dataset(self.folder_name_dataset)

    def load_tracklets_from_dataset(self, folder_name_dataset: str) -> DataFrame:
        """
        Load the whole prepared datasets from the folder_name_dataset with .csv files and save them in one DataFrame. 
        """

        self.load_dataframes(folder_name_dataset)        
            
        dataframe_all_trajectories = pd.concat(self.dataframes_dataset, ignore_index=True)
        self.dataframes_dataset = []

        # convert the Type field to the number
        dataframe_all_trajectories['Type'] = dataframe_all_trajectories.Type.astype('category').cat.codes
        return dataframe_all_trajectories
    
    def load_dataframes(self, folder_name: str) -> None:
        """
        Load all .csv files from a directory whose names contain the dataset token
        and store them as DataFrames in ``self.dataframes_dataset``.

        Args:
        folder_name (str): Path to the directory containing the .csv files.
        """

        for file_name in os.listdir(folder_name):
            if os.path.isfile(os.path.join(folder_name, file_name)):
                if self.MAGNI_DATASET_NAME_TOKEN in file_name:
                    self.load_dataframe(file_name)

    def load_dataframe(self, file_name: str, nrows: int = None):
        """
        Create the DataFrame() from .csv file.

        Args:
        file_name (str): Name of .csv file.
        """

        file_path = self.folder_name_dataset + '/' + file_name
        df = pd.read_csv(file_path, nrows=nrows)        
        df.drop(df.columns[0], axis = 1, inplace=True)          
        self.dataframes_dataset.append(df)

    def get_dataframe_all_trajectories(self) -> DataFrame:
        return self.dataframe_all_trajectories
    
    def __len__(self):
        return len(self.dataframe_all_trajectories)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        series_tracklet = self.dataframe_all_trajectories.iloc[idx, 1:]        
        list_tracklet = self.series_to_list(series_tracklet)
        input_dim = len(list_tracklet[0])
        array_tracklet = np.array([list_tracklet], dtype=float).reshape(-1, input_dim)

        label = self.dataframe_all_trajectories.iloc[idx, 0]
        sample = {'tracklet': array_tracklet, 'label': label}

        if self.transformation:
            sample = self.transformation(sample)

        return sample

    def series_to_list(self, series: str) -> list:
        """
        Transform the series '[123, 123]' to [123, 123]. 
        """
        series_string_list = [i for i in series]
        series_list = []

        for tracklet_point_string in series_string_list:
            tracklet_point_string = tracklet_point_string.replace('[', '')
            mapping_table = str.maketrans({'[': '', ']': '', ',': ''})
            tracklet_point_string = tracklet_point_string.translate(mapping_table)

            value_string_list = list(tracklet_point_string.split(" "))
            tracklet_point = [float(number) for number in value_string_list]
            series_list.append(tracklet_point)

        return series_list

class TrackletNormalization(object):
    """Normalize the tracklets to (0, 0) coordinates."""

    def __call__(self, sample):
        first_point_tracklet = np.tile(sample['tracklet'][0], (sample['tracklet'].shape[0], 1))
        normalised_tracklet = np.subtract(sample['tracklet'], first_point_tracklet)

        new_sample = {'tracklet': normalised_tracklet, 'label': sample['label']}

        return new_sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tracklet, label = sample['tracklet'], sample['label']

        return {'tracklet': torch.from_numpy(tracklet),
                'label': torch.tensor(label, dtype=torch.int8)}

if __name__ == "__main__":
    DATASET_FOLDER_NAME = 'tracklets_4s_4hz_v'

    dataset = TrackletsDataset(DATASET_FOLDER_NAME, transforms.Compose([TrackletNormalization(), ToTensor()]))