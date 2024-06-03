import logging
import copy
import torch
import torch.nn as nn
import numpy as np
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from inspect import signature


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def balance_data(self, data_list, param = None):
        # Convert all labels, which is less than threshold to zero
        label_array = np.array([label if label >= param else 0. for label in data_list[0]])
        # Sort labels and data
        ind = np.argsort(label_array)
        label_array = label_array[ind]
        data_list[1] = data_list[1][ind]
        data_list[0] = label_array
        # Count how many zero labels we have
        zero_number = np.count_nonzero(data_list[0]==0)
        # If we don't have zero labels the balancing doesn't make sense
        if zero_number == 0:
            return []
        # Create the arrays of zero labels and corresponding data
        label_array = data_list[0][0:zero_number]
        data_array = data_list[1][0:zero_number]
        # Create the arrays without zero labels and corresponding data
        label_array_wo0 = data_list[0][zero_number:]
        data_array_wo0 = data_list[1][zero_number:]
        # We can have labels with the value is higher than 1.0 . It can be social tracklets but we want to use it only if we don't have other option
        positive_indexs = np.where(label_array_wo0 <= 1.0)[0]
        # If we don't have enough labels which <1.0 we can just take zero number cases from the begginig of without zero labels
        if len(positive_indexs) < zero_number:
            label_array = np.concatenate((label_array, data_list[0][zero_number:2 * zero_number]), axis = 0)
            data_array = np.concatenate((data_array, data_list[1][zero_number:2 * zero_number]))
        # Otherwise we should take the data with the highest label values
        else:
            positive_indexs = positive_indexs[-zero_number:]
            label_array = np.concatenate((label_array, label_array_wo0[positive_indexs]), axis = 0)
            data_array = np.concatenate((data_array, data_array_wo0[positive_indexs]))
        # After that labels should be in the phorm 0.0 or 1.0
        label_array = np.array([0. if label == 0. else 1. for label in label_array])
        data_list = [label_array, data_array]

        return data_list

    def create_labels(self, data, tracklet_length = 16, param = None, agent = 'robot'):
        label_list = []
        # If the agent is human, we don't know the goal, so the goal will be the last point of the tracklet
        if agent == 'human':
            for i in range(len(data)):
                for j in range(len(data[i])):
                    data[i][j][4], data[i][j][5] = data[i][-1][0], data[i][-1][1]
        # Iteration through data (tracklet)
        for tracklet in data:
            # We need to worry about changing of the goal for the robot
            # Take the goal of the first point in the tracklet and assign to the last goal
            l_gx, l_gy = tracklet[0][4], tracklet[0][5]
            # Take the positions of the first point in the tracklet
            rx, ry = tracklet[0][0], tracklet[0][1]
            # Take the index to calculate how many points the current goal have 
            index_goal = len(tracklet) - 1
            # Iteration through the tracklet from the end to find the last point with the last goal
            for j in reversed(range(len(tracklet))):
                if l_gx == tracklet[j][4] and l_gy == tracklet[j][5]:
                    rx_lg, ry_lg = tracklet[j][0], tracklet[j][1]
                    index_goal = j
                    break
            # Calculate the distance between first and the last point for the current goal
            dist = ((rx - rx_lg)**2 + (ry - ry_lg)**2)**0.5
            # Initialize the list with states for each goal in the tracklet
            # State is included: the best dist between first and last points, the actual dist, the last index of the goal
            dist_list = [[dist, 0.0, index_goal]]
            # Initialize the counter of different goals in one tracklet
            counter_goal_update = 0
            # Iteration through tracklet (agent pose) by index
            for state_index in range(len(tracklet) - 1):
                # Take the positions
                rx, ry = tracklet[state_index][0], tracklet[state_index][1]
                # Take the next positions
                f_rx, f_ry = tracklet[state_index + 1][0], tracklet[state_index + 1][1]
                # Take the current goal
                gx, gy = tracklet[state_index][4], tracklet[state_index][5]
                # Check if the goal is different from the previous point
                if l_gx != gx or l_gy != gy:
                    # Change the last goal
                    l_gx, l_gy = gx, gy
                    # Iteration through the tracklet from the end to find the last point with the last goal
                    for j in reversed(range(len(tracklet))):
                        if l_gx == tracklet[j][4] and l_gy == tracklet[j][5]:
                            rx_lg, ry_lg = tracklet[j][0], tracklet[j][1]
                            index_goal = j
                            break
                    # Calculate the distance between the current point and the last point for the updated goal in the tracklet
                    dist = ((rx - rx_lg)**2 + (ry - ry_lg)**2)**0.5
                    # Update the counter of the goals in the tracklet
                    counter_goal_update += 1
                    # Add a new element in the list of goals
                    dist_list.append([dist, 0.0, index_goal])
                # Calculate the distance between the current  and the next points
                dist  = ((rx - f_rx)**2 + (ry - f_ry)**2)**0.5
                # Update the real distance for the current goal
                dist_list[counter_goal_update][1] += dist
            # Initialize the extra distance relation for the tracklet
            extra_dist_relat = 0
            # If the tracklet has only one goal just calculate the extra distance relation
            if len(dist_list) == 1:
                extra_dist_relat  += dist_list[0][0] / dist_list[0][1]
            else:
                # Else we calculate the extra distance relation for each goal, multiply them on the koef and sum
                previous_index = 0
                for j in dist_list:
                    # koef is the ratio of the number of points to one target to the number of points in the tracklet
                    koef = (j[2] - previous_index + 1) / tracklet_length
                    extra_dist_relat += j[0] / j[1] * koef
                    previous_index = j[2]
            # If we have param we need to convert the output to 0,1 , else we return the list of extra_dist_relat
            if param is None:
                label_list.append(extra_dist_relat)
            else:
                # Checking the extra distance relation condition for the tracklet and safe label for the next using during retraining
                if extra_dist_relat < param:
                    label_list.append(0.)
                else: label_list.append(1.)
        return np.array(label_list)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False, from_test = False, ol = False, start_episode = None):
        
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        dist_relat_list = []
        time_relat_list = []
        past_goal_list = []
        
        # The constant for additional classificator
        # The input tracklet for the additional classificator consists of the previous positions + the next act + linear prediction of future positions.
        # freq_time_mode + 1 + self.robot.freq_time_param * (1 - self.robot.soc_class_coef) = tracklet_len
        freq_time_mode = int(self.robot.freq_time_param * self.robot.soc_class_coef)

        # The constants for online learning 
        if ol:
            # We need to understand the length of input tracklets for the model. The second element of modules() is the first layer
            first_layers = [module for module in self.robot.classificator.model.modules()][1]
            # We can take hidden_size only from GRU layer
            if type(first_layers) == nn.GRU:
                tracklet_len = first_layers.hidden_size
            else: tracklet_len = 20
            check_update = 3
            coef_acc = 0.5
            soc_dist_th = 0.9
            soc_extra_dist_th = 0.9
            # Copy the model to have a basic model for each new start of simulation
            # model_for_train = copy.deepcopy(self.robot.classificator)
        for i in range(k):
            if start_episode is None:
                ob = self.env.reset(phase)
            else:
                ob = self.env.reset(phase, test_case = start_episode + i)

            # Add new goals to increse the time of the experiment in test, define new goals   
            if phase == 'test':       
                if self.env.movement_pattern == 'center':
                    # goals = [[0.0, -self.env.circle_radius],
                    #         [0.0, 0.0],
                    #         [self.env.circle_radius, 0.0],
                    #         [-self.env.circle_radius, 0.0],
                    #         [0.0, 0.0]]
                    # goals = [[0.0, -self.env.circle_radius],
                    #         [0.0, 0.0],
                    #         [self.env.circle_radius, 0.0],
                    #         [-self.env.circle_radius, 0.0],
                    #         [0.0, 0.0],
                    #         [0.0, -self.env.circle_radius],
                    #         [0.0, 0.0],
                    #         [0.0, self.env.circle_radius],
                    #         [0.0, 0.0],
                    #         [self.env.circle_radius, 0.0],
                    #         [-self.env.circle_radius, 0.0],
                    #         [0.0, 0.0]]
                    goals = [[0.0, -self.env.circle_radius],
                            [0.0, 0.0],
                            [self.env.circle_radius, 0.0],
                            [-self.env.circle_radius, 0.0],
                            [0.0, 0.0],
                            [0.0, -self.env.circle_radius]]
                else:
                    goals = []
                    px, py = self.robot.get_position()
                    gx, gy = self.env.next_circle_goal(px, py)
                    self.robot.gx, self.robot.gy = gx, gy
                    number_goals = int(self.env.circle_number * 360 / abs(self.env.step_degree))
                    for _ in range(number_goals - 1):
                        n_px, n_py = gx, gy
                        gx, gy = self.env.next_circle_goal(n_px, n_py, step_def = self.env.step_degree)
                        goals.append([gx, gy])  
            else:
                px, py = self.robot.get_position()
                # goals = [[px, py]]
                goals = []
            # print(goals)
            # goals = []
            # Add new goals to the robot object
            self.robot.clean_goals()
            for goal in goals:
                self.robot.add_goal(goal)             
            # Calculate the ideal distance for the robot way
            px, py, gx, gy = self.robot.px, self.robot.py, self.robot.gx, self.robot.gy
            dist_best = ((px - gx)**2 + (py - gy)**2)**0.5
            for j in self.robot.next_goal_list:
                px, py, gx, gy = gx, gy, j[0], j[1]
                dist_best += ((px - gx)**2 + (py - gy)**2)**0.5
            # Calculate the ideal time for the robot way
            time_best = dist_best/self.robot.v_pref
            # The variables which represente the real distance and time of the robot way
            time_real = 0
            dist_real = 0

            # The variables for online learning 
            if ol:
                robot_tracklet_list = []
                robot_tracklet_label_list = []
                human_tracklet_list = []
                human_tracklet_label_list = []
                # counter_new_tracklet = 0
                robot_tracklet_df = None
                # Copy the model to have a basic model for each new start of simulation
                model_for_train = copy.deepcopy(self.robot.classificator)
            
            done = False
            states = []
            actions = []
            rewards = []
            while not done:                
                # We want to have a list of the robot goal positions to calculate some metrics later 
                past_goal_list.append((self.robot.gx, self.robot.gy))
                # If the policy has additional classificator, we need to add the work with this classificator
                if (self.robot.policy.name == 'SARL-SOC') or ol:
                    # If the number of previous positions is enough we can form the tracklet
                    if len(self.env.states[-freq_time_mode + 2:]) > freq_time_mode - 3:
                        # Take freq_time_mode - 2 positions as a previous positions
                        pos = np.array([self.env.states[-j-1][0].position for j in reversed(range(len(self.env.states[-freq_time_mode + 2:])))])
                        vel = np.array([self.env.states[-j-1][0].velocity for j in reversed(range(len(self.env.states[-freq_time_mode + 2:])))])
                        tracklet = np.concatenate([pos, vel], 1)
                        # Take a current position
                        rp = self.robot.get_position()
                        rv = self.robot.get_velocity()
                        robot_array = np.array([[rp[0], rp[1], rv[0], rv[1]]])
                        tracklet = np.concatenate((tracklet, robot_array), 0)
                        # Take a position and linear prediction of future positions after next action
                        # If we have OL we need to predict the action based on the updated model
                        
                        if ol:
                            action = self.robot.act_test(ob, tracklet, model_for_train)
                        # If we don't have OL we are using the initial model
                        else: action = self.robot.act_test(ob, tracklet, self.robot.classificator)
                    else: action = self.robot.act(ob, dist_best, dist_real)               
                else:action = self.robot.act(ob, dist_best, dist_real)
                ob, reward, done, info = self.env.step(action, phase = phase)
                if type(action)==ActionRot:
                    # for nonholonomic
                    dist_real = dist_real + action.v*self.robot.time_step
                else:
                    # for holonomic
                    dist_real = dist_real + ((action.vx**2+action.vy**2)**0.5)*self.robot.time_step
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                
                # ________________________________________
                # Online learning module
                # print(self.robot.policy.name[-3:])
                # if (ol and self.robot.policy.name == 'SARL-SOC') or (ol and self.robot.policy.name != 'SARL-SOC'):
                if ol:
                    # After each step we try to find the full tracklet and if the number of full tracklets is enough we do the update of the weight
                    # if len(self.env.states) == tracklet_len * (counter_new_tracklet + 1):
                    if len(self.env.states) % tracklet_len == 0:
                        # Take last new robot tracklet and goals of robot
                        pos = np.array([self.env.states[-j-1][0].position for j in reversed(range(len(self.env.states[-tracklet_len:])))])
                        vel = np.array([self.env.states[-j-1][0].velocity for j in reversed(range(len(self.env.states[-tracklet_len:])))])
                        goal = np.array(past_goal_list[-tracklet_len:])
                        tracklet = np.concatenate([pos, vel, goal], 1)
                        robot_tracklet_list.append(tracklet)
                        # Take a last new human tracklets                        
                        pos_h = np.array([self.env.states[-j-1][1] for j in reversed(range(len(self.env.states[-tracklet_len:])))])
                        # Create a list of lists. The len equals the number of people
                        human_list =  [[] for j in range(len(pos_h[0]))]
                        # humans is the list of FullStates of all human per one time step 
                        for humans in pos_h:
                            # human is one FullState, human_l is the list, which will save the position and velocity from human FullState 
                            for human, human_l in zip(humans, human_list):
                                pos = human.position
                                vel = human.velocity
                                goal = human.goal_position
                                human_l.append(pos + vel + goal)
                        # Save the human tracklets to retrain the model later
                        for j in human_list:
                            human_tracklet_list.append(j)
                        # Increase the counter to take the last positions for a new tracklet
                        # counter_new_tracklet += 1

                        # If we have enough number of new tracklets we make a check for the model update
                        if int(len(robot_tracklet_list) % check_update) == 0:                                  

                                # # Create the list of soc criteon for each robot position in the tracklet
                                # robot_tracklet_soc_criterion = []
                                # # Choose the human tracklets by index, which correponds to the current robot tracklet
                                # humans_tracklet = human_tracklet_df[robot_tracklet_index * human_number:(robot_tracklet_index + 1) * human_number]
                                # # Choose the robot tracklet
                                # robot_tracklet = robot_tracklet_df[robot_tracklet_index]
                                # # Iteration through robot tracklet (robot pose) by index
                                # for state_index in range(len(robot_tracklet)):
                                #     rx, ry = robot_tracklet[state_index][0], robot_tracklet[state_index][1]
                                #     # Initialisation of the minimum distance between robot and people around
                                #     min_soc_dist = 10 * soc_dist_th
                                #     # Iteration through human tracklets (human tracklet) to find the minimum distance between robot and person
                                #     for human_tracklet in humans_tracklet:
                                #         hx, hy = human_tracklet[state_index][0], human_tracklet[state_index][1]
                                #         dist = ((rx - hx)**2 + (ry - hy)**2)**0.5
                                #         min_soc_dist = min(min_soc_dist, dist)
                                #     # Checking the social condition for the robot pose
                                #     if min_soc_dist < soc_dist_th:
                                #         robot_tracklet_soc_criterion.append(0)
                                #     else: robot_tracklet_soc_criterion.append(1)
                                # # Safe label of the robot tracklet for the next using during retraining
                                # robot_tracklet_label = 1 if int(np.mean(robot_tracklet_soc_criterion)) == 1 else 0
                                # robot_tracklet_label_list.append(robot_tracklet_label)
                            # print(robot_tracklet_label_list)

                            robot_data = np.array(robot_tracklet_list)
                            robot_tracklet_label_list = self.create_labels(robot_data, tracklet_length = tracklet_len, param = soc_extra_dist_th)
                                
                            # We need to use only last human tracklets
                            human_tracklet_df = np.asarray(human_tracklet_list[-check_update * len(human_list):])
                            # Update 20.04.2024
                            # human_tracklet_df = np.asarray(human_tracklet_list)
                            # We need to cut the goal of humans for social classificator
                            if human_tracklet_df.shape[-1] != 4:
                                human_tracklet_df = np.delete(human_tracklet_df, [4, 5], -1)
                            y_pred = self.robot.classificator.test(human_tracklet_df)
                            print(y_pred.numpy()[0])
                            y_pred = model_for_train.test(human_tracklet_df)
                            print(y_pred.numpy()[0])

                            human_data = np.array(human_tracklet_list[-check_update * len(human_list):])
                            # Update 20.04.2024
                            # human_data = np.array(human_tracklet_list)
                            human_label = self.create_labels(human_data, tracklet_length = tracklet_len, param = soc_extra_dist_th, agent = 'human')
                            human_label = torch.tensor(human_label).unsqueeze(0).float()
                            y_pred = y_pred.squeeze(0)
                            print(human_label.numpy()[0])
                            accuracy = (y_pred == human_label).sum().item() / y_pred.size(0)

                            # If the last human movements are not enough social, it means that the model works bad and we need to retrain the model
                            # The last human movements are labeled by extra_distance relation.
                            if accuracy < coef_acc:
                                print('Train')
                                # Prepare the human tracklets for retraining of the model
                                human_data = np.array(human_tracklet_list)

                                # All human tracklets are social
                                # human_label = np.ones(human_data.shape[0])

                                # Define the label of human tracklets by distance relation parameter
                                human_label = self.create_labels(human_data, tracklet_length = tracklet_len, param = soc_extra_dist_th, agent = 'human')

                                # We need to cut the goal of humans for social classificator
                                if human_data.shape[-1] != 4:
                                    human_data = np.delete(human_data, [4, 5], -1)

                                # The input for train
                                data_list = [human_label, human_data]

                                # Prepare the robot tracklets for retraining of the model
                                robot_data = []
                                robot_label = []
                                # We need to use for retrain only the tracklets, which are unsocial
                                for robot_tracklet_label, robot_tracklet in zip(robot_tracklet_label_list, robot_tracklet_list):
                                    if robot_tracklet_label == 0:
                                        robot_label.append(robot_tracklet_label)
                                        robot_data.append(robot_tracklet)
                                # If we have unsocial robot tracklets we can add them to the train input
                                if robot_data:
                                    robot_data = np.array(robot_data)
                                    # Just drop the goal's columns if we have them
                                    if robot_data.shape[-1] != 4:
                                        robot_data = np.delete(robot_data, [4, 5], -1)
                                    robot_label = np.array(robot_label)
                                    data_list[0] = np.concatenate((data_list[0], robot_label), axis = 0)
                                    data_list[1] = np.concatenate((data_list[1], robot_data))
                                # train_data
                                # Make the balanced sub-dataset, where the number of positive and negative cases equal
                                # data_list = self.balance_data(data_list, param = soc_extra_dist_th)
                                # If the prepared train dataset after the balance is empty we shouldn't train the model 
                                if data_list:
                                    print(data_list[0])
                                    model_for_train.train(data_list)
                                # Clean the list of human and robot tracklets so we would not retrain on the same data
                                robot_tracklet_list = []
                                robot_tracklet_label_list = []
                                human_tracklet_list = []
                            print()
                # ________________________________________
            
                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)                
            time_real = self.env.global_time
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                # Calculate the distance and time relations only for the success cases when the function is called from test.py
                if from_test:
                    time_relation = time_best/time_real
                    dist_relation = dist_best/dist_real
                    reward_final = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)])
                    extra_info = 'in episode {} '.format(i)
                    logging.info('{:<5} {}has time relation: {:.3f}, dist relation: {:.3f}, reward: {:.4f}, time {:.3f}'.
                     format(phase.upper(), extra_info, time_relation, dist_relation, reward_final, time_real))
                    dist_relat_list.append(dist_relation)
                    time_relat_list.append(time_relation)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        print(success_times)
        print(time_relat_list)
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, mean nav time: {:.2f}, median nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, np.median(success_times),
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('The mean extra time relation: %.3f, the median extra time relation: %.3f, and the extra distance relation: %.3f',
                         np.mean(time_relat_list),np.median(time_relat_list), np.mean(dist_relat_list))
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
                    # value = reward 
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
