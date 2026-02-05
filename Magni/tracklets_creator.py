import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from trajectory import Trajectory
from pandas import DataFrame

MAGNI_DATASET_FREQ = 100

class TrackletsCreator:
    """Class for transforming trajectories into tracklets from source data."""
    def __init__(self, source_data_folder_name: str = 'Clean_data', source_dataset_name: str = 'THOR-Magni'):
        """
        Load csv files names of the source dataset. 
        """
        source_code_folder_name = os.path.dirname(os.path.abspath(__file__))
        self.folder_path_source_data = os.path.join(source_code_folder_name, source_data_folder_name)
        self.source_dataset_name = source_dataset_name
        self.csv_files_names_source_data = []
        self.people_trajectories_source_data = []
        self.robot_trajectories_source_data = []
        self.time_counters_source_data = []

    def load_csv_names_source_data(self, folder_path_source_data: str = None) -> None:
        """
        Load csv files names of the source dataset. 
        """
        if folder_path_source_data is None:
            folder_path_source_data = self.folder_path_source_data
            
        for object_name in os.listdir(folder_path_source_data):
            object_full_path = os.path.join(folder_path_source_data, object_name)
            if os.path.isfile(object_full_path):
                if self.source_dataset_name in object_name:
                    self.csv_files_names_source_data.append(object_name)

    def csv_files_data_to_trajectories(self) -> None:
        """
        Extract the human and robot trajectories from all csv files of the source dataset.
        """
        for csv_file_name in self.csv_files_names_source_data:
            df = self.load_df_from_csv(csv_file_name) 

            # round the time to remove the time like 0.5600000000001
            df['Time'] = round(df['Time'], 2)           
            
            self.extract_trajectories_from_df(df)

    def load_df_from_csv(self, csv_file_name: str, nrows: int = None) -> DataFrame:
        """
        Create a DataFrame from a .csv file. 
        """
        csv_file_path = os.path.join(self.folder_path_source_data, csv_file_name)
        df = pd.read_csv(csv_file_path, nrows=nrows)

        # drop the first column with unnecessary information (Unnamed column)
        UNNECESSARY_COLUMN = 0
        df.drop(df.columns[UNNECESSARY_COLUMN], axis = 1, inplace=True)

        self.time_counters_source_data.append(df['Time'].unique())

        return df

    def extract_trajectories_from_df(self, df: DataFrame,
                                     people_color_list: list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                                                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) -> None:
        """
        Extract people and robot trajectories from a dataframe.
        """
        # collect all column names for people agents
        PERSON_KEYWORD = '_person_'
        person_columns = [column for column in df.columns if re.search(PERSON_KEYWORD, column)]

        PEOPLE_PHRASE_AFTER_ID = '_person_X'
        people_trajectories = self.extract_trajectories_from_columns(df, person_columns, PEOPLE_PHRASE_AFTER_ID, people_color_list)
        self.people_trajectories_source_data.append(people_trajectories)
        
        rob_idx_columns = ((len(df.columns) - 1) // 2) + 1
        robot_columns = df.columns[1 : rob_idx_columns]

        ROBOT_PHRASE_AFTER_ID = '_X'
        robot_color_list = ['#000000']
        robot_trajectories = self.extract_trajectories_from_columns(df, robot_columns, ROBOT_PHRASE_AFTER_ID, robot_color_list)
        self.robot_trajectories_source_data.append(robot_trajectories)

    def extract_trajectories_from_columns(self, df: DataFrame, columns: list[str], phrase_after_id: str = '', trajectories_color = []) -> list[Trajectory]:
        """
        Extract and save trajectories from df DataFrame with columns names.
        """
        trajectories = []

        agents = self.columns_to_agents(columns)

        X_AGENT_IDX = 0
        Y_AGENT_IDX = 1
        TIME_COLUMN = 'Time'
        for i in agents:
            id = i[X_AGENT_IDX].replace(phrase_after_id, '')
            frame = df[TIME_COLUMN].values
            x = df[i[X_AGENT_IDX]].values
            y = df[i[Y_AGENT_IDX]].values

            if not np.isnan(x).all():
                trajectory = Trajectory(id, frame, x, y, trajectories_color)
                trajectories.append(trajectory)

        return trajectories

    def columns_to_agents(self, column_names: list) -> list[list]:
        """
        Convert the list of column names to the list of agents. 
        """
        return [[column_names[2*i], column_names[2*i + 1]] for i in range(int(len(column_names)/2))]
    
    def convert_trajectories_to_tracklets(self, tracklet_points_number: int = 4, tracklet_frequency_hz: int = MAGNI_DATASET_FREQ,
                                            tracklet_csv_folder: str = 'tracklets') -> None:
        """
        Convert all robot and human trajectories to tracklets. Save tracklets as csv files. 
        """        
        if self.people_trajectories_source_data and self.robot_trajectories_source_data:
            # the basic datasets include points at each 1/MAGNI_DATASET_FREQ second (MAGNI_DATASET_FREQ hz)
            if tracklet_frequency_hz != MAGNI_DATASET_FREQ:
                self.change_people_robots_trajectories_frequency(tracklet_frequency_hz)

            columns_names_tracklets_dataframe = self.create_column_names_for_tracklets(tracklet_points_number)

            for people_trajectories_from_one_file, robot_trajectories_from_one_file in zip(self.people_trajectories_source_data, self.robot_trajectories_source_data):
                # initialize the data frame with tracklets for the database
                tracklets_df = pd.DataFrame(columns=columns_names_tracklets_dataframe)
                tracklets_df = self.transform_trajectories_to_tracklets(tracklets_df, people_trajectories_from_one_file, 'People')
                tracklets_df = self.transform_trajectories_to_tracklets(tracklets_df, robot_trajectories_from_one_file, 'Robot')

                # save the df with tracklets like .csv file
                name_index = self.people_trajectories_source_data.index(people_trajectories_from_one_file)
                self.save_tracklets_df_as_csv(self.csv_files_names_source_data[name_index], tracklets_df, tracklet_csv_folder)

    def change_people_robots_trajectories_frequency(self, new_frequency_hz: int) -> None:
        """
        Delete the points from people and robot trajectories based on new frequency. The original frequency is 100. 
        """
        new_time_step = 1 / new_frequency_hz
        
        for people_trajectories_list, robot_trajectories_list in zip(self.people_trajectories_source_data, self.robot_trajectories_source_data):  
            self.change_trajectories_new_time_step(people_trajectories_list, new_time_step)
            self.change_trajectories_new_time_step(robot_trajectories_list, new_time_step)
                
    def change_trajectories_new_time_step(self, trajectory_list: list[Trajectory], new_time_step: int) -> None:
        """
        Delete the points from trajectories based on new time step. The original time step is 0.01 . 
        """
        for trajectory in trajectory_list:
            # initialise the start time of any trajectory
            time = 0.01
            x_new, y_new, frames_new = [], [], []
            while (time <= trajectory.frames[-1]):
                time_idx = trajectory.frames.index(time)

                x_new.append(trajectory.x[time_idx])
                y_new.append(trajectory.y[time_idx])
                frames_new.append(time)

                time = round(time + new_time_step, 2) 
            
            trajectory.x, trajectory.y = x_new, y_new
            trajectory.frames = frames_new

    def create_column_names_for_tracklets(self, tracklet_length_in_points: int) -> list:
        """
        Create a list of column names for the future Dataframe with tracklets.
        The first column 'Type' is responsible for the type of the tracklet: the tracklet was created from a human or a robot trajectory.
        """
        first_column = 'Type'
        columns_names = [first_column]

        for i in range(tracklet_length_in_points):
            columns_names.append('Point_' + str(i + 1))

        return columns_names
    
    def transform_trajectories_to_tracklets(self, df_tracklets: DataFrame, trajectories: list[Trajectory], trajectories_type_name: str) -> DataFrame:
        """
        Transform trajectories to tracklets. Add the velocities of tracklet's points.  
        """   
        columns_names_tracklets_dataframe = list(df_tracklets)
        tracklet_points_number = len(columns_names_tracklets_dataframe) - 1

        for trajectory in trajectories:
            start_tracklet_index = 0

            while (start_tracklet_index + tracklet_points_number < len(trajectory.x)):
                # choose the sub-trajectory
                x_tracklet = trajectory.x[start_tracklet_index : start_tracklet_index + tracklet_points_number]
                y_tracklet = trajectory.y[start_tracklet_index : start_tracklet_index + tracklet_points_number]                
                trajectory_time = trajectory.frames

                # if the sub trajectory doesn't include nan values create a tracklet
                if not np.isnan(x_tracklet).any():
                    tracklet = [[x_point_value, y_point_value] for x_point_value, y_point_value in zip(x_tracklet, y_tracklet)]

                    # check if the tracklet is not a repeatable point or nan
                    tracklet_repeatable = True
                    for i in tracklet[1:]:
                        if tracklet[0] != i:
                            tracklet_repeatable = False

                    if not tracklet_repeatable:
                        # calculate velocity and add it to the tracklet
                        # TODO: recalculate velocities of first point of a tracklet
                        Vx_point_value, Vy_point_value = 0, 0
                        tracklet[0].append(Vx_point_value)
                        tracklet[0].append(Vy_point_value)

                        point_index = 1
                        while point_index < len(tracklet):
                            dt = trajectory_time[point_index] - trajectory_time[point_index - 1]
                            Vx_point_value = (tracklet[point_index][0] - tracklet[point_index - 1][0])/dt
                            Vy_point_value = (tracklet[point_index][1] - tracklet[point_index - 1][1])/dt  
                            tracklet[point_index].append(Vx_point_value)
                            tracklet[point_index].append(Vy_point_value)
                            point_index += 1

                        new_row = {'Type': trajectories_type_name}
                        for column_name, point_tracklet in zip(columns_names_tracklets_dataframe[1:], tracklet):
                            new_row[column_name] = point_tracklet
                        df_tracklets = pd.concat([df_tracklets, pd.DataFrame([new_row])], ignore_index=True)
                start_tracklet_index += tracklet_points_number

        return df_tracklets

    def save_tracklets_df_as_csv(self, file_name: str, tracklets_df: DataFrame, folder_name: str = 'tracklets') -> None:
        """
        Save the tracklets Dataframe as a csv.file in folder_name with the file_name.
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        tracklets_csv_name = folder_name + '/tracklets_' + file_name

        tracklets_df.to_csv(tracklets_csv_name, na_rep=np.nan)

        print(tracklets_csv_name + ' is saved.')

    def animate_trajectories(self, df_index: int, robot_index: int, legend: bool = False, only_robots: bool = False) -> None:
        """
        Draw the plot of the persons trajectories and robot_index robot trajectory from df_index DataFrame. 
        """
        fig, ax = plt.subplots()
        ax.set(xlim=(-9500, 9500), ylim=(-5000, 5000))

        # the lists of lines coordinates and lines objects
        line_coordinates = []
        lines = []
        
        # create the object for the time counter (text)
        text_kwargs = dict(ha = 'right', va = 'top', fontsize = 10, color = 'black')
        text = ax.text(9500, 5000, '', **text_kwargs, animated = True)

        # save the list of trajectories which will be drawn
        if only_robots:
            trajectory_list = self.robot_trajectories_source_data[df_index]
            ax.set(xlim=(-14500, 14500), ylim=(-10000, 10000))
            for tr in trajectory_list:
                color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                tr.color = color_list[tr.id % len(color_list)]
        else:
            trajectory_list = self.people_trajectories_source_data[df_index]
            trajectory_list.append(self.robot_trajectories_source_data[df_index][robot_index])

        # create a line object for each trajectory
        for traject in trajectory_list:
            (line,) = ax.plot([], [], color = traject.color, animated = True, label = str(traject.id))
            lines.append(line)
            line_coordinates.append([[],[]])

        if legend:
            ax.legend()

        plt.show(block=False)
        plt.pause(0.1)
        bg = fig.canvas.copy_from_bbox(fig.bbox) 
        for line in lines:
            ax.draw_artist(line)
        fig.canvas.blit(fig.bbox)

        for frame in self.time_counters_source_data[df_index]:
            fig.canvas.restore_region(bg)
            for traj_num in range(len(trajectory_list)):
                # if the frame of last trajectory point is bigger then the current frame plus threshold -> the trajectory is still observed
                coord = trajectory_list[traj_num].get_coord_by_frame(frame)
                if coord[0] is not None:
                    line_coordinates[traj_num][0].append(coord[0])
                    line_coordinates[traj_num][1].append(coord[1])
                # else we need to do the trajectory empty, so it will not be drawn. Also we clean the plot and redraw the axes 
                lines[traj_num].set_data(line_coordinates[traj_num][0], line_coordinates[traj_num][1])
                ax.draw_artist(lines[traj_num])
            # draw the current frame
            frame_string = 'Frame: ' + str(frame)
            text.set_text(frame_string)
            ax.draw_artist(text)
             # update the animation
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()      

            plt.pause(0.0001)

    def plot_trajectories(self, dataframe_idx: int, robot_trajectory_idx: int) -> None:
        """
        Draw all people trajectories and a robot trajectory with index robot_trajectory_idx from dataframe_idx Dataframe. 
        """
        trajectories = self.people_trajectories_source_data[dataframe_idx]
        trajectories.append(self.robot_trajectories_source_data[dataframe_idx][robot_trajectory_idx])

        for trajectory in trajectories:
            trajectory.plot_traject()
        
        plt.show()
            
if __name__ == "__main__":

    trainer = TrackletsCreator()
    trainer.load_csv_names_source_data()
    trainer.csv_files_data_to_trajectories()

    # trainer.animate_trajectories(0, 1, legend = True)
    # trainer.plot_trajectories(0, 0)

    time = 4
    hz = 4
    steps = time * hz
    folder_name = 'tracklets_' + str(time) + 's_' + str(hz) + 'hz_v'
    trainer.convert_trajectories_to_tracklets(tracklet_points_number = steps, tracklet_frequency_hz = hz, tracklet_csv_folder = folder_name)