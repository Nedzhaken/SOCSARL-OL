import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import re
from datetime import datetime
import os
import math
from collections import namedtuple

# Import for ORCA policy
from pyorca import Agent, orca, normalized
from halfplaneintersect import InfeasibleError

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])

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

class Agent_sl(object):
    """A disk-shaped agent"""
    def __init__(self, goal_position, position, velocity, radius, max_speed, pref_velocity, holonomic):
        self.goal_position = goal_position
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.max_speed = max_speed
        self.pref_velocity = pref_velocity
        self.kinematics = holonomic
        self.speed_samples = 5
        self.rotation_samples = 17
        self.build_action_space(pref_velocity)
        
        # Parameters, which can be placed in the config file.
        # If the agents are closer then this distance, it makes sense to check the collision.
        self.dist_threshold = 5
        # If the agent is closer to the goal then this distance, the goal is reached.
        self.reach_threshold = self.radius
        # Time step of the simulation.
        self.dt = 0.01

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            # rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
            rotations = np.linspace(-np.pi, np.pi, self.rotation_samples, endpoint=True)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)
        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def action_check(self, collider, action):
        """Check the collision with other agent."""
        # Calculate the position of collider in t + 1.
        next_pos_col = [collider.position[0] + collider.velocity[0]*self.dt, collider.position[1] + collider.velocity[1]*self.dt]
        # Calculate the position of agent with the choosed action in t + 1.
        next_pos_agent = [self.position[0] + action.vx*self.dt, self.position[1] + action.vy*self.dt]
        # If will not be collision return False, else return True.
        dist = math.dist(next_pos_col, next_pos_agent)
        return True if dist<(self.radius + collider.radius) else False

    def action_space_remove_by_speed(self, action_space, value = [0, 0]):
        """Remove from the action space the action with the velocity from value."""
        for action in action_space:
            if action.vx == value[0] and action.vy == value[1]:
                action_space.remove(action)
                break

    def act(self, ob = []):
        """Choose the action (velocity). ob is the list of agents around."""
        action_space = self.action_space.copy()
        # Iterate each agent around.
        for agent in ob:
            # If there is no observation, we should not worry about this agent.
            if not np.isnan(agent.position[0]):
                # If the agent is further than dist_threshold the collision cannot be possible.
                if math.dist(self.position, agent.position)<self.dist_threshold:
                    # Iterate each action.
                    for action in self.action_space:
                        check = self.action_check(agent, action)
                        # If the action leads to collision, we will remove this action from the list.
                        if check:
                            action_space.remove(action)
                    pass
        # If there are not actions, the agent is in the collision. The agent just stay.
        if not action_space:
            return ActionXY(0, 0)
        else:
            # Calculate the current distance between agent and goal positions.
            current_goal_dist = math.dist(self.position, self.goal_position)
            return_action = ActionXY(0, 0)
            # Iterate throw each action to find the action, which leads to closest position to the goal.
            for action in action_space:
                # Predict the next agent position.
                next_pos_agent = [self.position[0] + action.vx*self.dt, self.position[1] + action.vy*self.dt]
                # Calculate the distance to the goal from the future position.
                new_goal_dist = math.dist(next_pos_agent, self.goal_position)
                # If the new future position is closer to the goal, update the action.
                if new_goal_dist<current_goal_dist:
                    return_action = action
                    current_goal_dist = new_goal_dist
            return return_action
    
    def step(self, action):
        """Update the agent parameters"""
        self.position = [self.position[0] + action.vx*self.dt, self.position[1] + action.vy*self.dt]
        self.velocity = [action.vx, action.vy]

    def is_goal_reach(self):
        """Check the goal is reached or not"""
        # dist = math.dist(self.position, self.goal_position)
        dist = math.sqrt(((self.position[0] - self.goal_position[0])**2) + ((self.position[1] - self.goal_position[1])**2))
        return True if dist<self.reach_threshold else False

class Simulation:
    def __init__(self):
        # The file name of .csv file
        folder_name = 'THOR-Magni_ICRA_Subset'
        sub_folder_name = 'human_motion'
        file_name = 'THOR-Magni_130522_SC2_R2.csv'
        self.full_name = folder_name + '/' + sub_folder_name + '/' + file_name
        # The database from .csv file
        self.df = None
        # The list of database trajectories
        self.traject_list = []
        # The list of robot agent, trajectory and robot trajectory database
        self.robot = None
        self.traject_robot = None
        self.robot_tr_df = None
        # The unique frames, which will be used like a time of the simulation
        self.time_uniq = None
        # The axis limits for animation
        self.maxX = float('-inf')
        self.maxY = float('-inf')
        self.minX = float('inf')
        self.minY = float('inf')

    def set_db_file_name(self, name):
        """
        Set the full_name of the simulation. This name is used to read the database with people trajectories.
        """
        self.full_name = name

    def add_robot(self, goal, pos, vel, rad, max_speed, pref_velocity, holonomic, time = None):
        """
        Create the robot agent. 
        """
        self.robot = Agent_sl(goal, pos, vel, rad, max_speed, pref_velocity, holonomic)
        # Choose the start time of the robot's movement and create time list before this time.
        if time:
            dt = self.traject_list[0].frames[1] - self.traject_list[0].frames[0]
            time_list = np.arange(0, time, dt)
        else:
            time_list = []
        # Fill the robot trajectory list with nan before start time.
        traject_robot_x = [np.nan for t in time_list]
        traject_robot_y = [np.nan for t in time_list]
        # Create the global robot trajectory and add the curent robot's position.
        self.traject_robot = Trajectory(0, np.array(self.traject_list[0].frames),
                                        np.array(pos[0]), np.array(pos[1]))
        traject_robot_x.append(self.traject_robot.x)  
        traject_robot_y.append(self.traject_robot.y)
        self.traject_robot.x = traject_robot_x
        self.traject_robot.y = traject_robot_y
        # Choose the black color for the robot trajectory.
        self.traject_robot.color = '#000000'

    def calculate_robots_trajectory(self, save = False):
        """
        Calculate robot's trajectory. 
        """
        # if the robot object exists
        if self.robot:
            # Create the list of observation
            ob_list = []
            for trajectory in self.traject_list:
                # Create the agent for each trajectory
                goal = [trajectory.x[-1], trajectory.y[-1]]
                pos = [trajectory.x[0], trajectory.y[0]]
                vel = [0, 0]
                rad = 0.3
                max_speed = self.robot.max_speed
                pref_velocity = self.robot.pref_velocity
                holonomic = 'holonomic'
                ob = Agent_sl(goal, pos, vel, rad, max_speed, pref_velocity, holonomic)
                ob.dt = trajectory.frames[1] - trajectory.frames[0]
                ob_list.append(ob)
            # Calculate the robot's trajectory
            frame_index = 0
            while not self.robot.is_goal_reach() and frame_index < len(self.time_uniq)-1:                            
                action = self.robot.act(ob_list)
                self.robot.step(action)
                for trajectory, ob in zip(self.traject_list, ob_list):
                    velocityX = (trajectory.x[frame_index + 1] - trajectory.x[frame_index]) / ob.dt
                    velocityY = (trajectory.y[frame_index + 1] - trajectory.y[frame_index]) / ob.dt
                    action = ActionXY(velocityX, velocityY)
                    ob.step(action)
                frame_index = frame_index + 1
                self.traject_robot.x.append(self.robot.position[0])
                self.traject_robot.y.append(self.robot.position[1])
            diff_len = len(self.time_uniq) - len(self.traject_robot.x)
            self.traject_robot.x =  self.traject_robot.x + [self.traject_robot.x[-1]] * diff_len
            self.traject_robot.y =  self.traject_robot.y + [self.traject_robot.y[-1]] * diff_len
            # Save the robot's trajectory
            if save:
                self.safe_robot_trajectory()

    def calculate_robots_trajectory_orca(self, save = False):
        """
        Calculate robot's trajectory which is controlled by orca policy. 
        """
        # if the robot object exists
        if self.robot:
            # Create the list of observation
            ob_list = []
            # Create the list of orca_agents
            orca_agents = []
            for trajectory in self.traject_list:
                # Create the agent for each trajectory
                goal = [trajectory.x[-1], trajectory.y[-1]]
                pos = [trajectory.x[0], trajectory.y[0]]
                vel = [0, 0]
                rad = 300
                max_speed = self.robot.max_speed
                pref_velocity = self.robot.pref_velocity
                holonomic = 'holonomic'
                ob = Agent_sl(goal, pos, vel, rad, max_speed, pref_velocity, holonomic)
                ob.dt = trajectory.frames[1] - trajectory.frames[0]
                ob_list.append(ob)

                # Create the orca_agent for each trajectory
                # pref_velocity is important only for robot. For observation we put zero array.
                orca_agents.append(Agent((pos[0], pos[1]), (0., 0.), rad, max_speed, np.array(vel)))

            # Find the preferred velocity to go to the goal of the robot
            theta = math.atan2(self.robot.position[1]-self.robot.goal_position[1], self.robot.position[0]-self.robot.goal_position[0])           
            x = np.array((np.cos(theta), np.sin(theta)))
            try:
                pref_velocity = normalized(-x) * self.robot.max_speed
            except AssertionError:
                pref_velocity = np.array([0., 0.])
            # Create orca agent object
            robot_orca = Agent((float(self.robot.position[0]), float(self.robot.position[1])),
            (self.robot.velocity[0], self.robot.velocity[1]), self.robot.radius, self.robot.max_speed, pref_velocity)
            # Defined the time parameters
            tau = 2
            dt = self.robot.dt            
            frame_index = 0
            robot_start_index = len(self.traject_robot.x) - 1
            # Calculate the robot's trajectory
            while not self.robot.is_goal_reach() and frame_index < len(self.time_uniq)-1:
                # Start update the robot's position and trajectory only after robot's start time 
                if frame_index >= robot_start_index:
                    # Create a new orca_agents list with only existed orca_agent (position != nan)
                    orca_agents_sort = [orca_agent for orca_agent in orca_agents if not math.isnan(orca_agent.position[0])]
                    try:
                        new_vels, __ = orca(robot_orca, orca_agents_sort, tau, dt)
                    except InfeasibleError:
                        new_vels = np.array([0., 0.])
                    except AssertionError:
                        new_vels = np.array([0., 0.])
                    # ORCA can not find the solution and return nan. In this case we need to change the values to 0.
                    if np.isnan(new_vels[0]):
                        new_vels = np.array([0., 0.])
                    # If ORCA returns values, which are more than limit speed we need to limit it.
                    if np.linalg.norm(new_vels) > self.robot.max_speed:
                    # if any(new_vels > self.robot.max_speed):
                        koef = np.arctan2(new_vels[1], new_vels[0])
                        new_vels[0] = self.robot.max_speed * np.cos(koef)
                        new_vels[1] = self.robot.max_speed * np.sin(koef)
                    # Update ORCA robot agent state
                    robot_orca.velocity = new_vels
                    robot_orca.position += robot_orca.velocity * dt
                    # Update robot agent state
                    action = ActionXY(new_vels[0], new_vels[1])
                    self.robot.step(action)
                    # Update ORCA robot agent pref_velocity
                    theta = math.atan2(self.robot.position[1]-self.robot.goal_position[1], self.robot.position[0]-self.robot.goal_position[0])           
                    x = np.array((np.cos(theta), np.sin(theta)))
                    try:
                        robot_orca.pref_velocity = normalized(-x) * self.robot.max_speed
                    except AssertionError:
                        robot_orca.pref_velocity = np.array([0., 0.])
                    # Save new robot positions in the robot trajectories
                    self.traject_robot.x.append(self.robot.position[0])
                    self.traject_robot.y.append(self.robot.position[1])
                # Update people agents state
                for trajectory, ob in zip(self.traject_list, ob_list):
                    velocityX = (trajectory.x[frame_index + 1] - trajectory.x[frame_index]) / ob.dt
                    velocityY = (trajectory.y[frame_index + 1] - trajectory.y[frame_index]) / ob.dt
                    action = ActionXY(velocityX, velocityY)
                    ob.step(action)
                # Update people orca agents state
                for trajectory, o_a in zip(self.traject_list, orca_agents):
                    velocityX = (trajectory.x[frame_index + 1] - trajectory.x[frame_index]) / dt
                    velocityY = (trajectory.y[frame_index + 1] - trajectory.y[frame_index]) / dt
                    o_a.velocity = (velocityX, velocityY)
                    o_a.position = (trajectory.x[frame_index + 1], trajectory.y[frame_index + 1])
                # Update frame counter
                frame_index = frame_index + 1
            # Fill the trajectory by last point.
            diff_len = len(self.time_uniq) - len(self.traject_robot.x)
            self.traject_robot.x =  self.traject_robot.x + [np.nan] * diff_len
            self.traject_robot.y =  self.traject_robot.y + [np.nan] * diff_len
            # Save the robot's trajectory
            if save:
                self.safe_robot_trajectory()

    def update_robot_goal(self, goal):
        """
        Change the robot goal. The robot trajectory will be saved.  
        """
        # If the robot object exists.
        if self.robot:
            # Update the goal.
            self.robot.goal_position = goal
            # Clean the robot trajectory from np.nan on the end of the trajectory (if the previous goal was reached).
            while self.traject_robot.x and np.isnan(self.traject_robot.x[-1]):
                self.traject_robot.x.pop()
                self.traject_robot.y.pop()
            # Update the goal.
            self.robot.position = [self.traject_robot.x[-1], self.traject_robot.y[-1]]
            pass

    def safe_robot_trajectory(self, folder = None):
        """
        Save the robot trajectory. 
        """
        # Check if we have what we want to save.
        if self.traject_robot:
            # Check if we saved somethong before. If not we should create the file name.
            if self.robot_tr_df is None:
                self.robot_tr_df = pd.DataFrame(data={'Time': self.time_uniq})
                index = self.full_name.rfind('.csv')
                now = datetime.now()
                self.robot_tr_df_name = self.full_name[:index] + '_robot_path_' + now.strftime("%d-%m-%Y_%H_%M_%S") + '.csv'
            # Id to separate the different trajectories.
            id_robot_trajectory = str((len(self.robot_tr_df.columns) - 1) // 2)
            self.robot_tr_df[id_robot_trajectory + '_X'] = self.traject_robot.x
            self.robot_tr_df[id_robot_trajectory + '_Y'] = self.traject_robot.y
            # Check if we want to save this file in separate folder.
            if folder:
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                index = self.robot_tr_df_name.rfind('/') + 1
                file_name = folder + self.robot_tr_df_name[index - 1:]
                self.robot_tr_df.to_csv(file_name, na_rep=np.nan)
            else:
                self.robot_tr_df.to_csv(self.robot_tr_df_name, na_rep=np.nan)

    def safe_people_trajectory(self, folder = None):
        """
        Save the people trajectories, which was downloaded from the self.full_name . 
        """
        # Check if we have what we want to save.
        if self.traject_list:
            # Check if we saved somethong before. If not we should create the file name.
            if self.robot_tr_df is None:
                self.robot_tr_df = pd.DataFrame(data={'Time': self.time_uniq})
                index = self.full_name.rfind('.csv')
                now = datetime.now()
                self.robot_tr_df_name = self.full_name[:index] + '_robot_path_' + now.strftime("%d-%m-%Y_%H_%M_%S") + '.csv'
            for traject in self.traject_list:
                # Id to separate the different trajectories.
                id_person_trajectory = str(traject.id)
                self.robot_tr_df[id_person_trajectory + '_person_X'] = traject.x
                self.robot_tr_df[id_person_trajectory + '_person_Y'] = traject.y
            # Check if we want to save this file in separate folder.
            if folder:
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                index = self.robot_tr_df_name.rfind('/') + 1
                file_name = folder + self.robot_tr_df_name[index - 1:]
                self.robot_tr_df.to_csv(file_name, na_rep=np.nan)
            else:
                self.robot_tr_df.to_csv(self.robot_tr_df_name, na_rep=np.nan)

    def load_db(self, nrows=None):
        """
        Create the DataFrame() from .csv file. 
        """
        # Load only three lines of the databease to understand which helmets were used
        # Because of THOR-Magni_300922_SC5_R4 file, we need to check all files if there are the Helmet with 0 markers.
        df = pd.read_csv(self.full_name, nrows=2, skiprows=12)
        # The second line we don't need
        df = df.drop([0])
        # Drop all not helmet columns
        useless_columns = [s for s in df.columns if not re.search('Helmet_', s)]
        df.drop(useless_columns, axis = 1, inplace=True)
        # Find which helmet has 0 marks
        useless_helmets = [s for s in df.columns if df[s].values[0] == '0']
        useless_helmets_str = ''
        for s in useless_helmets:
            useless_helmets_str = useless_helmets_str + '|' + s

        # Load the databease by the name of .csv file
        df = pd.read_csv(self.full_name, nrows=nrows, skiprows=16)
        # Clean the database. We need only centroid coordinates of the helmets
        useless_columns = [s for s in df.columns if re.search('DARKO_Robot|LO1' + useless_helmets_str, s)]
        df.drop(useless_columns, axis = 1, inplace=True)
        useless_columns = [s for s in df.columns if not re.search('Frame|Time|Centroid_X|Centroid_Y', s)]
        df.drop(useless_columns, axis = 1, inplace=True)
        # Because of THOR-Magni_170522_SC3A_R2 file, we need to sort the data by frames and create the missing time slots for them.
        df = df.sort_values(by=['Frame'])
        df['Time'] = (df['Frame'] - 1) * 0.01
        # Drop all duplicates rows (1700 and 1701 for THOR-Magni_170522_SC3A_R2)
        df = df.drop_duplicates(subset=['Frame'])
        df = df.reset_index(drop=True)

        # Fill the missed data
        df = df.interpolate(method='linear', limit_direction='forward', axis=0)
        
        self.df = df
        # Save time for the simulation
        self.time_uniq = self.df['Time'].unique()

    def db_to_traj(self, traj_numb = 10000000, color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']):
        """
        Add all trajectories from the current database to the list of trajectories (traject_list).
        traj_numb is the number of trajectories which will be taken from the database.
        """
        self.traject_list = []
        # Create the list of helmet names
        columns = [s for s in self.df.columns if re.search('Helmet_', s)]
        helmet_list = [[columns[2*i], columns[2*i + 1]] for i in range(int(len(columns)/2))]
        for i in helmet_list:
            id = i[0].replace('Helmet_', '').replace(' Centroid_X', '')
            frame = self.df['Time'].values
            x = self.df[i[0]].values
            y = self.df[i[1]].values

            # if the trajectory exists but includes only nan values.
            if not np.isnan(x).all():
                traject = Trajectory(id, frame, x, y, color_list)
                self.traject_list.append(traject)

    def generate_trajectories(self, step = 100, folder = None):
        """
        Generate the robot trajectories, which is similar to person trajectory. For each person creates one robot trajectory.
        """
        # Create the robot trajectory by each person trajectory.
        for person_trajectory in self.traject_list:
            # Find the start time moment.
            st_time_index = 0
            while np.isnan(person_trajectory.x[st_time_index]):
                st_time_index += 1
            # Create the robot agent. It will start from the start time of the person trajectory.
            pos = [person_trajectory.x[st_time_index], person_trajectory.y[st_time_index]]
            goal = [0, 0]
            vel = [0, 0]
            rad = 300
            max_speed = 500
            pref_velocity = max_speed
            holonomic = 'holonomic'
            if st_time_index == 0:
                self.add_robot(goal, pos, vel, rad, max_speed, pref_velocity, holonomic)
            else:
                self.add_robot(goal, pos, vel, rad, max_speed, pref_velocity, holonomic, time = person_trajectory.frames[st_time_index])
            step = step
            # Update the goal of robot with the step. It is needed to simulate the movements parallel to human movements.
            while st_time_index + step < len(person_trajectory.x):
                new_goal = [person_trajectory.x[st_time_index + step], person_trajectory.y[st_time_index + step]]              
                self.update_robot_goal(new_goal)
                self.calculate_robots_trajectory_orca(save = False)
                st_time_index += step
            self.safe_robot_trajectory(folder)
            n = 6000
            plt.plot(self.traject_robot.x[0:n], self.traject_robot.y[0:n])
            plt.plot(person_trajectory.x[0:n], person_trajectory.y[0:n])
            plt.show()
        self.safe_people_trajectory(folder)

    def plot_trajectories(self):
        """
        Draw the plot of the current trajectories. 
        """
        for i in self.traject_list:
            i.plot_traject()
        plt.show()

    def animate_trajectories(self, legend = False):
        """
        Draw the plot of the current trajectories during increasing of the frames. 
        """
        # create a figure and axis
        fig, ax = plt.subplots()
        ax.set(xlim=(-9500, 9500), ylim=(-5000, 5000))

        # The lists of trajectory coordinates and lines objects 
        line_coord_list = []
        ln_list = []
        
        # Create the object for the frame counter (text)
        text_kwargs = dict(ha='right', va='top', fontsize=10, color='black')
        text = ax.text(9500, 5000, '', **text_kwargs, animated=True)
        # Add the robot trajectory, if the robot was added
        if self.robot:
            self.traject_list.append(self.traject_robot)
        # Create the line object for each trajectory
        for traject in self.traject_list:
            (ln,) = ax.plot([], [], color = traject.color, animated=True, label = str(traject.id))
            ln_list.append(ln)
            line_coord_list.append([[],[]])

        # Draw the legend 
        if legend: ax.legend()

        plt.show(block=False)
        plt.pause(0.1)
        bg = fig.canvas.copy_from_bbox(fig.bbox)  
        for ln in ln_list: ax.draw_artist(ln)
        fig.canvas.blit(fig.bbox)

        # Iterate by each frame (time)
        for frame in self.time_uniq:
            fig.canvas.restore_region(bg)
            for traj_num in range(len(self.traject_list)):
                # If the frame of last trajectory point is bigger then the current frame plus threshold -> the trajectory is still observed
                if self.traject_list[traj_num].frames[-1] > (frame + 1):
                    coord = self.traject_list[traj_num].get_coord_by_frame(frame)
                    if coord[0] != None:
                        line_coord_list[traj_num][0].append(coord[0])
                        line_coord_list[traj_num][1].append(coord[1])
                # Else we need to do the trajectory empty, so it will not be drawn. Also we clean the plot and redraw the axes 
                ln_list[traj_num].set_data(line_coord_list[traj_num][0], line_coord_list[traj_num][1])
                ax.draw_artist(ln_list[traj_num])
            # Draw the current frame
            frame_string = 'Frame: ' + str(frame)
            text.set_text(frame_string)
            ax.draw_artist(text)
             # Update the animation
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()      

            plt.pause(0.001)
        # If the robot was added, clean the people trajectory
        if self.robot:
            self.traject_list.pop()

    def start_simulation(self):
        """
        Start the simulation. 
        """
        self.load_db()
        self.db_to_traj()
        self.generate_trajectories(step = 400, folder = 'Clean_data')

        # pos = [-7100, 3000]
        # goal = [-900, -3550]
        # vel = [0, 0]
        # rad = 300
        # max_speed = 500
        # pref_velocity = max_speed
        # holonomic = 'holonomic'
        # self.add_robot(goal, pos, vel, rad, max_speed, pref_velocity, holonomic, time = 0.01)
        # # self.calculate_robots_trajectory(save = True)
        # self.calculate_robots_trajectory_orca(save = False)
        # new_goal = [900, 3550]
        # self.update_robot_goal(new_goal)
        # self.calculate_robots_trajectory_orca(save = False)
        # self.animate_trajectories(legend=True)

# # directory/folder path
# dir_path = 'THOR-Magni_ICRA_Subset/human_motion'

# # list to store files
# res = []

# # Iterate Scenario directories
# for path in os.listdir(dir_path):
#     # Check if the current folder is a folder with databases
#     if 'Scenario' in path:
#         # Create full path to the folder with databases
#         full_path = os.path.join(dir_path, path)
#         # Iterate database files
#         for file_name in os.listdir(full_path):
#             # Create full path to the file with database
#             file_path = os.path.join(full_path, file_name)
#             # Check if current file_path is a file
#             assert os.path.isfile(file_path), 'The folder should contain only files'
#             # Create a short path of database file 
#             file_name = os.path.join(path, file_name)
#             file_name = file_name.replace("\\", "/")

#             # add filename to list                      
#             res.append(file_name)

# print(res)
# for file_name_db in res:
#     simulation = Simulation()
#     index = simulation.full_name.rfind('/') + 1
#     name = simulation.full_name[:index] + file_name_db
#     print(name)
#     print()
#     simulation.set_db_file_name(name)
#     simulation.start_simulation()

# simulation = Simulation()
# simulation.set_db_file_name('THOR-Magni_ICRA_Subset/human_motion/Scenario_5/THOR-Magni_300922_SC5_R4.csv')
# simulation.start_simulation()
# simulation.plot_trajectories()

# print('END')