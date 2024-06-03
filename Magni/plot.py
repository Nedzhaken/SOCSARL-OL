import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
import re
from read_Magni_dataset import Trajectory
from PIL import Image
import img2pdf

class Drawer:
    def __init__(self):
        """
        The class to transform our trajectories to tracklets. 
        """
        # The name of folder with prepared dataset .csv file.
        self.folder_name = 'Clean_data'
        # The database from .csv file.
        self.df = None
        # The list of .csv file names.
        self.csv_names = None
        # The list of tracklet databases.
        self.df_tr_list = []
        # The list of database trajectories.
        self.traject_ped_list = []
        self.traject_rob_list = []
        # The unique frames, which will be used like a time of the simulation.
        self.time_uniq = []

    def load_db(self, name, nrows=None):
        """
        Create the DataFrame() from .csv file. 
        """
        file_name = self.folder_name + '/' + name
        df = pd.read_csv(file_name, nrows=nrows)        
        df.drop(df.columns[0], axis = 1, inplace=True)    
        
        self.df = df
        # Save time for the simulation
        self.time_uniq.append(self.df['Time'].unique())

    def db_to_traj(self, color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']):
        """
        Add all trajectories from the current database to the list of trajectories (traject_list).
        traj_numb is the number of trajectories which will be taken from the database.
        """
        # Choose person columns.
        person_columns = [s for s in self.df.columns if re.search('_person_', s)]
        # Choose robot columns.
        rob_index_columns = ((len(self.df.columns) - 1) // 2) + 1
        robot_columns = self.df.columns[1:rob_index_columns]   
        # Update the list of persons trajectories.
        traject_list = []
        person_list = [[person_columns[2*i], person_columns[2*i + 1]] for i in range(int(len(person_columns)/2))]
        # Round the time to remove the time like 0.5600000000001.
        self.df['Time'] = round(self.df['Time'], 2)
        # Save the db information like Trajectories for persons.
        for i in person_list:
            id = i[0].replace('_person_X', '')
            frame = self.df['Time'].values
            x = self.df[i[0]].values
            y = self.df[i[1]].values

            # if the trajectory exists but includes only nan values.
            if not np.isnan(x).all():
                traject = Trajectory(id, frame, x, y, color_list)
                traject_list.append(traject)
        self.traject_ped_list.append(traject_list)

        # Update the list of robots trajectories
        traject_list = []
        rob_list = [[robot_columns[2*i], robot_columns[2*i + 1]] for i in range(int(len(robot_columns)/2))]
        # Save the db information like Trajectories for robot.
        for i in rob_list:
            id = i[0].replace('_X', '')
            frame = self.df['Time'].values
            x = self.df[i[0]].values
            y = self.df[i[1]].values

            # if the trajectory exists but includes only nan values.
            if not np.isnan(x).all():
                traject = Trajectory(id, frame, x, y, color_list)
                traject.color = '#000000'
                traject_list.append(traject)
        self.traject_rob_list.append(traject_list)

    def load_csv_from_folder(self, folder = None, file_name = None):
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
            if file_path == file_name:
                # check if current file_path is a file
                if os.path.isfile(os.path.join(dir_path, file_path)):
                    if 'THOR-Magni' in file_path:
                        # add filename to list
                        self.csv_names.append(file_path)
        for file_name_db in self.csv_names:
            self.load_db(file_name_db)
            self.db_to_traj()

    def plot_pair_trajectoies(self, ped_traj, rob_traj, st = 0, end = -1, index = 0, save = False):
        fontTimes = "Times New Roman"
        size = 24
        fontsize_boost = -5

        fig, ax = plt.subplots()
        X_r = [i / 1000 for i in rob_traj[index].x[st:end]]
        Y_r = [i / 1000 for i in rob_traj[index].y[st:end]]
        X_h = [i / 1000 for i in ped_traj[index].x[st:end]]
        Y_h = [i / 1000 for i in ped_traj[index].y[st:end]]
            
        ax.plot(X_r, Y_r, label='Part of generated robot trajectory')
        ax.plot(X_h, Y_h, label='Part of participant trajectory')
        # index = index -1 
        # X_r = [i / 1000 for i in rob_traj[index].x[st:end]]
        # Y_r = [i / 1000 for i in rob_traj[index].y[st:end]]
        # X_h = [i / 1000 for i in ped_traj[index].x[st:end]]
        # Y_h = [i / 1000 for i in ped_traj[index].y[st:end]]
            
        # ax.plot(X_r, Y_r, label='Part of generated robot trajectory')
        # ax.plot(X_h, Y_h, label='Part of participant trajectory')
        ax.set_xlabel('X, Meters', fontsize=size+fontsize_boost - 8)
        ax.set_ylabel('Y, Meters', fontsize=size+fontsize_boost - 8)
        # ax.set_ylim(0, 100)
        font = font_manager.FontProperties(family=fontTimes, style='normal', size=size + fontsize_boost - 4)
        ax.tick_params(axis='x', labelsize=size + fontsize_boost - 8)
        ax.tick_params(axis='y', labelsize=size + fontsize_boost - 8)
        ax.legend( prop=font)
        ax.grid(True)
        plt.yticks(fontname = "Times New Roman")
        plt.xticks(fontname = "Times New Roman")
        # plt.show()
        if save:
            dir_path = 'Picture'
            if not os.path.exists(dir_path): os.makedirs(dir_path)
            fig.savefig(dir_path + '/' + 'Plot.png', dpi=300)

            img = Image.open(dir_path + '/' + 'Plot.png') 
            print(img.size)
            
            left = 50
            # top = 50
            top = 150
            right = 1870
            bottom = 1440            
            
            img_res = img.crop((left, top, right, bottom))
            img_res.show() 
            img.close()
            img_res.save(dir_path + '/' + 'Plot.png')

            img = Image.open(dir_path + '/' + 'Plot.png')

            pdf_path = dir_path + '/' + 'Plot.pdf'
            pdf_bytes = img2pdf.convert(img.filename)

            # opening or creating pdf file
            file = open(pdf_path, "wb")
            
            # writing pdf files with chunks
            file.write(pdf_bytes)

            # closing image file
            img.close()
            
            # closing pdf file
            file.close()

trainer = Drawer()
file_name = 'THOR-Magni_300922_SC5_R4_robot_path_16-01-2024_14_42_57.csv'
trainer.load_csv_from_folder(file_name = file_name)
num_traj = 1
trainer.plot_pair_trajectoies(trainer.traject_ped_list[0], trainer.traject_rob_list[0], index = num_traj, st = 0, end = 6000, save = True)
