import os
import pandas as pd
import pytest
from Magni.src.tracklets_creator import TrackletsCreator

@pytest.fixture
def track_creator():
    """Basic TrackletsCreator instance."""
    return TrackletsCreator()

@pytest.fixture
def track_creator_loaded(track_creator):
    """TrackCreator with CSV file names loaded."""
    folder_path_source_data = os.path.join(os.path.dirname(__file__), "test_data", "Clean_data")
    track_creator.load_csv_names_source_data(folder_path_source_data)
    return track_creator

@pytest.fixture
def track_creator_trajectories(track_creator_loaded):
    """TrackCreator with CSV file data converted to trajectories."""
    track_creator_loaded.csv_files_data_to_trajectories()
    return track_creator_loaded

def test_load_csv_names_source_data(track_creator_loaded):
    assert len(track_creator_loaded.csv_files_names_source_data) == 1
    assert track_creator_loaded.csv_files_names_source_data[0] == "THOR-Magni_120522_SC1A_R1_robot_path_16-01-2024_13_12_28.csv"

def test_csv_files_data_to_trajectories(track_creator_trajectories):
    assert len(track_creator_trajectories.people_trajectories_source_data[0]) == 9
    assert len(track_creator_trajectories.robot_trajectories_source_data[0]) == 9

def test_convert_trajectories_to_tracklets(track_creator_trajectories):
    test_folder_path = os.path.dirname(__file__)
    time = 4
    hz = 4
    steps = time * hz
    folder_name = os.path.join(test_folder_path, f"test_data/tracklets_{time}s_{hz}hz_v")

    track_creator_trajectories.convert_trajectories_to_tracklets(
        tracklet_points_number=steps,
        tracklet_frequency_hz=hz,
        tracklet_csv_folder=folder_name
    )

    df = pd.read_csv(os.path.join(folder_name, "tracklets_THOR-Magni_120522_SC1A_R1_robot_path_16-01-2024_13_12_28.csv"))
    df.drop(df.columns[0], axis=1, inplace=True)

    assert len(df) == 1030