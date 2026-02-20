import pytest
from Magni.src.trajectory import Trajectory
import numpy as np

class TestTrajectory:
    def setup_method(self):
        self.id = 0
        self.frames = np.array([0.01, 0.02, 0.03])
        self.x = np.array([2.0, 3.0, 4.0])
        self.y = np.array([6.0, 7.0, 8.0])
        self.trajectory = Trajectory(self.id, self.frames, self.x, self.y)

    def test_get_traject_frames(self):
        assert self.trajectory.get_traject_frames() == self.frames.tolist()

    def test_get_traject_x(self):
        assert self.trajectory.get_traject_x() == self.x.tolist()

    def test_get_traject_y(self):
        assert self.trajectory.get_traject_y() == self.y.tolist()

    @pytest.mark.parametrize(
    "frame, coordinates",
    [
        (0.01, [2.0, 6.0]),
        (0.07, [None, None]),
    ],
    )

    def test_get_coord_by_frame(self, frame, coordinates):
        assert self.trajectory.get_coord_by_frame(frame) == coordinates

    def test_get_coord_by_frame_default_val(self):
        default_value = 1
        assert self.trajectory.get_coord_by_frame(0.08, default_value) == [default_value, default_value]