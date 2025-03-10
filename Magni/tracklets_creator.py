import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from trajectory import Trajectory

class TrackletsCreator:
    def __init__(self):
        """
        The class to transform our trajectories to tracklets. 
        """
        # the name of folder with prepared dataset .csv file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.folder_name = os.path.join(script_dir, 'Clean_data')
        # the database from .csv file
        self.df = None
        # the list of .csv file names
        self.csv_names = None
        # the list of tracklet databases
        self.df_tr_list = []
        # the list of database trajectories
        self.traject_ped_list = []
        self.traject_rob_list = []
        # the unique frames, which will be used like a time of the simulation
        self.time_uniq = []

    def load_db(self, name, nrows=None):
        """
        Create the DataFrame() from .csv file. 
        """
        file_name = self.folder_name + '/' + name
        df = pd.read_csv(file_name, nrows=nrows)        
        df.drop(df.columns[0], axis = 1, inplace=True)    
        
        self.df = df
        # save time for the simulation
        self.time_uniq.append(self.df['Time'].unique())

    def db_to_traj(self, color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']):
        """
        Add all trajectories from the current database to the list of trajectories (traject_list).
        traj_numb is the number of trajectories which will be taken from the database.
        """
        # choose person columns
        person_columns = [s for s in self.df.columns if re.search('_person_', s)]
        # choose robot columns
        rob_index_columns = ((len(self.df.columns) - 1) // 2) + 1
        robot_columns = self.df.columns[1:rob_index_columns]   
        # update the list of persons trajectories
        traject_list = []
        person_list = [[person_columns[2*i], person_columns[2*i + 1]] for i in range(int(len(person_columns)/2))]
        # round the time to remove the time like 0.5600000000001
        self.df['Time'] = round(self.df['Time'], 2)
        # save the db information like Trajectories for persons
        for i in person_list:
            id = i[0].replace('_person_X', '')
            frame = self.df['Time'].values
            x = self.df[i[0]].values
            y = self.df[i[1]].values

            # if the trajectory exists but includes only nan values
            if not np.isnan(x).all():
                traject = Trajectory(id, frame, x, y, color_list)
                traject_list.append(traject)
        self.traject_ped_list.append(traject_list)

        # update the list of robots trajectories
        traject_list = []
        rob_list = [[robot_columns[2*i], robot_columns[2*i + 1]] for i in range(int(len(robot_columns)/2))]
        # save the db information like Trajectories for robot
        for i in rob_list:
            id = i[0].replace('_X', '')
            frame = self.df['Time'].values
            x = self.df[i[0]].values
            y = self.df[i[1]].values

            # if the trajectory exists but includes only nan values
            if not np.isnan(x).all():
                traject = Trajectory(id, frame, x, y, color_list)
                traject.color = '#000000'
                traject_list.append(traject)
        self.traject_rob_list.append(traject_list)

    def animate_trajectories(self, df_index, rob_index, legend = False, robots = False):
        """
        Draw the plot of the persons trajectories and rob_index robot trajectoris from df_index database. 
        """
        # create a figure and axis
        fig, ax = plt.subplots()
        ax.set(xlim=(-9500, 9500), ylim=(-5000, 5000))

        # the lists of trajectory coordinates and lines objects
        line_coord_list = []
        ln_list = []
        
        # create the object for the frame counter (text)
        text_kwargs = dict(ha='right', va='top', fontsize=10, color='black')
        text = ax.text(9500, 5000, '', **text_kwargs, animated=True)

        # save the list of trajectories which will be drawn
        if robots:
            traject_list = self.traject_rob_list[df_index]
            ax.set(xlim=(-14500, 14500), ylim=(-10000, 10000))
            for tr in traject_list:
                color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                tr.color = color_list[tr.id % len(color_list)]
        else:
            traject_list = self.traject_ped_list[df_index]
            traject_list.append(self.traject_rob_list[df_index][rob_index])

        # create the line object for each trajectory
        for traject in traject_list:
            (ln,) = ax.plot([], [], color = traject.color, animated=True, label = str(traject.id))
            ln_list.append(ln)
            line_coord_list.append([[],[]])

        # draw the legend
        if legend: ax.legend()

        plt.show(block=False)
        plt.pause(0.1)
        bg = fig.canvas.copy_from_bbox(fig.bbox) 
        for ln in ln_list: ax.draw_artist(ln)
        fig.canvas.blit(fig.bbox)

        # iterate by each frame (time)
        for frame in self.time_uniq[df_index]:
            fig.canvas.restore_region(bg)
            for traj_num in range(len(traject_list)):
                # if the frame of last trajectory point is bigger then the current frame plus threshold -> the trajectory is still observed
                coord = traject_list[traj_num].get_coord_by_frame(frame)
                if coord[0] != None:
                    line_coord_list[traj_num][0].append(coord[0])
                    line_coord_list[traj_num][1].append(coord[1])
                # else we need to do the trajectory empty, so it will not be drawn. Also we clean the plot and redraw the axes 
                ln_list[traj_num].set_data(line_coord_list[traj_num][0], line_coord_list[traj_num][1])
                ax.draw_artist(ln_list[traj_num])
            # draw the current frame
            frame_string = 'Frame: ' + str(frame)
            text.set_text(frame_string)
            ax.draw_artist(text)
             # update the animation
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()      

            plt.pause(0.0001)

    def plot_trajectories(self, df_index, rob_index, robots = False):
        """
        Draw the plot of the current trajectories. 
        """
        # save the list of trajectories which will be drawn
        if robots:
            traject_list = self.traject_rob_list[df_index]
        else:
            traject_list = self.traject_ped_list[df_index]
            traject_list.append(self.traject_rob_list[df_index][rob_index])

        for i in traject_list:
            i.plot_traject()
        plt.show()

    def cut_trajectory_hz(self, hz):
        """
        Delete the points from trajectories based on hz. The original hz is 100. 
        """
        # choose the time step
        time_step = 1 / hz
        # iteration by different files.
        for person_traj_list, robot_traj_list in zip(self.traject_ped_list, self.traject_rob_list):  
            # iterations by different trajectories
            for trajectory in person_traj_list:
                # initialise the start time 
                time = 0.01
                x_new, y_new, frames_new = [], [], []
                # iterations by each points of the trajectory
                while (time <= trajectory.frames[-1]):
                    # find the index of the point for the time
                    index = trajectory.frames.index(time)
                    # add x, y and time value to the lists
                    x_new.append(trajectory.x[index])
                    y_new.append(trajectory.y[index])
                    frames_new.append(time)
                    # update time
                    time = round(time + time_step, 2) 
                # update the trajectory
                trajectory.x, trajectory.y = x_new, y_new
                trajectory.frames = frames_new

            # iterations by different trajectories
            for trajectory in robot_traj_list:
                # initialise the start time      
                time = 0.01
                x_new, y_new, frames_new = [], [], []
                # iterations by each points of the trajectory
                while (time <= trajectory.frames[-1]):
                    # find the index of the point for the time
                    index = trajectory.frames.index(time)
                    # add x, y and time value to the lists
                    x_new.append(trajectory.x[index])
                    y_new.append(trajectory.y[index])
                    frames_new.append(time)
                    # update time
                    time = round(time + time_step, 2) 
                # update the trajectory
                trajectory.x, trajectory.y = x_new, y_new
                trajectory.frames = frames_new

    def create_tracklets(self, step = 4, hz = 100, save = False, folder = 'tracklets', velocity = False):
        """
        Transform all robot and human trajectories to tracklets. Save it. 
        """        
        # transform the trajectories if they exist
        if self.traject_ped_list and self.traject_rob_list:
            # the basic datasets include points at each 0.01 second (100 hz)
            # we can decrease the number of points for more realistic application
            if hz != 100:
                self.cut_trajectory_hz(hz)
            # initialize the column names of data frame with tracklets
            columns_names = ['Type']
            for i in range(step):
                columns_names.append('Point_' + str(i + 1))
            # iteration by different files
            for person_traj_list, robot_traj_list in zip(self.traject_ped_list, self.traject_rob_list):
                # initialize the data frame with tracklets for the database
                df_tr = pd.DataFrame(columns=columns_names)
                # iterations by different trajectories
                for trajectory in person_traj_list:
                    index = 0
                    # split the trajectory to the tracklets
                    while (index + step < len(trajectory.x)):
                        # choose the sub-trajectory
                        x = trajectory.x[index:index + step]
                        y = trajectory.y[index:index + step]
                        if velocity:
                            frames = trajectory.frames
                        # if the sub trajectory doesn't include nan values create a tracklet
                        if not np.isnan(x).any():
                            tracklet = [[x_value, y_value] for x_value, y_value in zip(x, y)]
                            # check if the tracklet is not a repeatable point or nan
                            flag_repeat = False
                            for i in tracklet[1:]:
                                if tracklet[0] != i: flag_repeat = True
                            # save the tracklet
                            if flag_repeat:                                
                                # calculate velocity and add it to the tracklet if the velocity flag is set
                                if velocity:
                                    # initialise the velocity of the first tracklets point
                                    Vx, Vy = 0, 0
                                    tracklet[0].append(Vx)
                                    tracklet[0].append(Vy)
                                    point_index = 1
                                    while point_index < len(tracklet):
                                        dt = frames[point_index] - frames[point_index - 1]
                                        Vx = (tracklet[point_index][0] - tracklet[point_index - 1][0])/dt
                                        Vy = (tracklet[point_index][1] - tracklet[point_index - 1][1])/dt  
                                        tracklet[point_index].append(Vx)
                                        tracklet[point_index].append(Vy)                                      
                                        point_index += 1
                                new_row = {'Type':'People'}
                                for name, point in zip(columns_names[1:], tracklet):
                                    new_row[name] = point
                                df_tr = df_tr.append(new_row, ignore_index=True)
                        index += step
                # iterations by different trajectories
                for trajectory in robot_traj_list:
                    index = 0
                    # split the trajectory to the tracklets
                    while (index + step < len(trajectory.x)):
                        # choose the sub-trajectory
                        x = trajectory.x[index:index + step]
                        y = trajectory.y[index:index + step]
                        if velocity:
                            frames = trajectory.frames
                        # if the sub trajectory doesn't include nan values create a tracklet
                        if not np.isnan(x).any():
                            tracklet = [[x_value, y_value] for x_value, y_value in zip(x, y)]
                            # check if the tracklet is not a repeatable point or nan
                            flag_repeat = False
                            for i in tracklet[1:]:
                                if tracklet[0] != i: flag_repeat = True
                            # save the tracklet.
                            if flag_repeat:
                                # calculate velocity and add it to the tracklet if the velocity flag is set
                                if velocity:
                                    # initialise the velocity of the first tracklets point
                                    Vx, Vy = 0, 0
                                    tracklet[0].append(Vx)
                                    tracklet[0].append(Vy)
                                    point_index = 1
                                    while point_index < len(tracklet):
                                        dt = frames[point_index] - frames[point_index - 1]
                                        Vx = (tracklet[point_index][0] - tracklet[point_index - 1][0])/dt
                                        Vy = (tracklet[point_index][1] - tracklet[point_index - 1][1])/dt  
                                        tracklet[point_index].append(Vx)
                                        tracklet[point_index].append(Vy)                                      
                                        point_index += 1
                                new_row = {'Type':'Robot'}
                                for name, point in zip(columns_names[1:], tracklet):
                                    new_row[name] = point
                                df_tr = df_tr.append(new_row, ignore_index=True)
                        index += step 
                # save the df with tracklets like .csv file
                if save:
                    name_index = self.traject_ped_list.index(person_traj_list)
                    self.save_tracklets(self.csv_names[name_index], df_tr, folder)

    def save_tracklets(self, file_name, df_tr, folder = 'tracklets'):
        # create the folder for tracklets
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # save df tracklet
        name = folder + '/tracklets_' + file_name
        df_tr.to_csv(name, na_rep=np.nan)
        print(name + ' is saved.')

    def load_csv_from_folder(self, folder = None):
        """
        Load the whole prepared dataset. 
        """
        # directory/folder path
        if folder == None:
            dir_path = self.folder_name
        else:
            dir_path = folder
        # list to store files
        self.csv_names = []
        # Iterate directory
        for file_path in os.listdir(dir_path):
            # check if current file_path is a file
            if os.path.isfile(os.path.join(dir_path, file_path)):
                if 'THOR-Magni' in file_path:
                    # add filename to list
                    self.csv_names.append(file_path)
        for file_name_db in self.csv_names:
            self.load_db(file_name_db)
            self.db_to_traj()
            
trainer = TrackletsCreator()
trainer.load_csv_from_folder()
time = 4
hz = 4
steps = time * hz
folder_name = 'tracklets_' + str(time) + 's_' + str(hz) + 'hz_v'
trainer.create_tracklets(step = steps, hz = hz, save = True, folder = folder_name, velocity = True)
