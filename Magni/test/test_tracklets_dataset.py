import os
import pytest
import numpy as np
import torch
from torchvision import transforms
from Magni.src.tracklets_dataset import TrackletsDataset, TrackletNormalization, ToTensor

def test_tracklet_normalization():
    transform = TrackletNormalization()

    sample = {
        "tracklet": np.array([[2., 3.],
                              [4., 6.],
                              [5., 7.]]),
        "label": 1
    }

    result = transform(sample)

    expected = np.array([[0., 0.],
                         [2., 3.],
                         [3., 4.]])

    assert np.array_equal(result["tracklet"], expected)
    assert result["label"] == 1

@pytest.fixture
def track_creator_totensor():
    """Provides a ToTensor transform instance."""
    return ToTensor()

def test_tracklet_dtype_preserved(track_creator_totensor):

    sample = {
        "tracklet": np.array([[1., 2.]], dtype=np.float64),
        "label": 0
    }

    result = track_creator_totensor(sample)

    assert result["tracklet"].dtype == torch.float64

def test_shape_preserved(track_creator_totensor):

    sample = {
        "tracklet": np.random.rand(10, 3),
        "label": 1
    }

    result = track_creator_totensor(sample)

    assert result["tracklet"].shape == (10, 3)

@pytest.fixture
def track_creator_dataset():
    """Provides a TrackletsDataset instance initialized from the
    test tracklets directory with normalization and tensor transforms applied."""
    script_folder_name = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER_NAME = script_folder_name + "/test_data/tracklets_4s_4hz_v"
    test_dataset = TrackletsDataset(DATASET_FOLDER_NAME, transforms.Compose([TrackletNormalization(), ToTensor()]))
    return test_dataset

def test_data_row_number(track_creator_dataset):
    assert track_creator_dataset.dataframe_all_trajectories.size == 17510