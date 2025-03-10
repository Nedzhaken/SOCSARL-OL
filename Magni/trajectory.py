import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self, id, frames, x, y, color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']):
        self.id = int(id)
        self.frames = frames.tolist()
        self.x = x.tolist()
        self.y = y.tolist()
        self.color = color_list[self.id % len(color_list)]

    def get_coord_by_frame(self, frame, default = None):
        """
        Get the list with coordinates X, Y of the person in the frame. If the trajectory doesn't have this frame, return [None, None]
        """
        try:
            index = self.frames.index(frame)
            return [self.x[index], self.y[index]]
        except ValueError:
            return [default, default]

    def get_traject_color(self):
        """
        Return the trajectory color. 
        """
        return self.color

    def plot_traject(self):
        """
        Draw the trajectory. 
        """
        plt.plot(self.x, self.y)

    def __str__(self):
        state_str = 'ID: ' + str(self.id) + '\n'
        for i in range(len(self.frames)):
            frame_str = 'Frame: {:5.1f} '.format(self.frames[i])
            coord_str = 'Coord: {:5.4f} {:5.4f}\n'.format(self.x[i], self.y[i])
            state_str = state_str +  frame_str + coord_str
        return state_str