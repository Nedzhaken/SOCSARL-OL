import torch
import torch.nn as nn
import numpy as np
import copy
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.policy.orca import ORCA
import math


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.steps = 3
        self.time_step = 0.25
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        # print('forward')
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        # e-vector
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        # h-vector
        mlp2_output = self.mlp2(mlp1_output)
        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL_CLASS(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL-SOC'
        self.soc_coef = 0.6
        logging.info('The social coef: %f', self.soc_coef)

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl_classif', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl_classif', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl_classif', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl_classif', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl_classif', 'with_om')
        with_global_state = config.getboolean('sarl_classif', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.movement_predictor = ORCA()
        print(self.model)
        self.multiagent_training = config.getboolean('sarl_classif', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL-SOC'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
    
    # def actionXY_to_actionROT(self, action):
    #     alpha = math.atan2(action.vy, action.vx)
    #     v = action.vx / math.cos(alpha)
    #     r = alpha - self.env.robot.theta
    #     action = ActionRot(v, r)
    #     return action
    
    def predict_class(self, state, classificator, tracklet, freq_time_param, dist_best = None, dist_real = None):
        """
        A base function, that takes pairwise joint state as input to value network. Also, it receives the tracklet and the social classificator.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """        
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        occupancy_maps = None
        probability = np.random.random()
        y_pred_best = None
        pred_list = []
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None

            actions_tensor = None
            rewards_tensor = None
            social_array = None
            # Calculate the max reward from the all actions
            for action in self.action_space:
                # Recalculate the new state of the robot after the action
                next_self_state = self.propagate(state.self_state, action)
                # Make a copy of this state, to calculate the linear prediction
                next_self_state_copy = copy.deepcopy(next_self_state)
                # Create the list of the new state.
                # The first element is the new state after action, and next states are the states after the linear prediction of the current action. 
                next_point_list = [[next_self_state_copy.px, next_self_state_copy.py, next_self_state_copy.vx, next_self_state_copy.vy]]
                # How many elements we need to predict
                size_difference = freq_time_param - tracklet.shape[0] - 1
                # In the case of non-holonomic movements we need to change r velocity 
                if self.kinematics == 'holonomic':
                    copy_action = ActionXY(action.vx, action.vy)
                else: copy_action = ActionRot(action.v, 0)
                # Make a linear prediction of the robot positions using the current action
                for _ in range(size_difference):
                    next_self_state_copy = self.propagate(next_self_state_copy, copy_action)
                    next_point_list.append([next_self_state_copy.px, next_self_state_copy.py, next_self_state_copy.vx, next_self_state_copy.vy])
                # # _________________________________
                # # Make an ORCA prediction of the next size_difference steps
                # next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                #                     for human_state in state.human_states]
                # next_full_state = JointState(next_self_state_copy, next_human_states)
                # # human_actions = [ActionXY(human_state.vx, human_state.vy) for human_state in state.human_states]
                # # next_obser_state = [human.get_next_observable_state(action) for human, action in zip(next_human_states, human_actions)]
                # for _ in range(size_difference):
                #     copy_action = self.movement_predictor.predict(next_full_state)
                #     # if self.kinematics != 'holonomic': copy_action = self.actionXY_to_actionROT(copy_action)
                #     if self.kinematics != 'holonomic': copy_action = self.env.robot.actionXY_to_actionROT(copy_action)
                #     next_self_state_copy = self.propagate(next_self_state_copy, copy_action)
                #     next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                #                     for human_state in next_human_states]
                #     next_full_state = JointState(next_self_state_copy, next_human_states)
                    
                #     next_point_list.append([next_self_state_copy.px, next_self_state_copy.py, next_self_state_copy.vx, next_self_state_copy.vy])
                # # _________________________________
                # Concatenate the previous robot positions, the future position from the action and the predicted positions
                next_poit_array = np.array(next_point_list)
                new_tracklet = np.concatenate((tracklet, next_poit_array), axis=0)
                new_tracklet = np.expand_dims(new_tracklet, axis=0)

                # Calculate the next states of the humans and the reward for the action
                if self.query_env:
                    # The choosing between two reward functions
                    if dist_real is None: next_human_states, reward, _, _ = self.env.onestep_lookahead(action)
                    else: next_human_states, reward, _, _ = self.env.onestep_lookahead(action, [dist_best, dist_real])
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                    for human_state in state.human_states]
                    # The choosing between two reward functions
                    if dist_real is None: reward = self.compute_reward(next_self_state, next_human_states)
                    else: reward = self.compute_reward(next_self_state, next_human_states, [dist_best, dist_real, action])
                reward = torch.Tensor([[reward]]).to(self.device)

                # Unify the robot state vector with humans state vector
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                            for next_human_state in next_human_states], dim=0)
                # The output of rotate in the agent-centric coordinate: dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)

                if actions_tensor is None:
                    actions_tensor = rotated_batch_input
                    rewards_tensor = reward
                    social_array = new_tracklet
                else:
                    actions_tensor = torch.cat((actions_tensor, rotated_batch_input))
                    rewards_tensor = torch.cat((rewards_tensor, reward))
                    social_array = np.concatenate((social_array, new_tracklet), axis=0)

            # VALUE UPDATE
            next_state_value = self.model(actions_tensor)
            values_tensor = rewards_tensor + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value  
            # Cummulate sum
            soc_pred_tensor = classificator.test(social_array).T
            if torch.mean(soc_pred_tensor).item() != 0.0 and torch.mean(soc_pred_tensor).item()!= 1.0:
                print(torch.mean(soc_pred_tensor).item())
            values_tensor = values_tensor + self.soc_coef * soc_pred_tensor
            self.action_values = torch.reshape(values_tensor, (-1,)).tolist()
            max_action = self.action_space[torch.argmax(values_tensor).data.item()]

            # # Calculate the max reward from the all actions
            # for action in self.action_space:
            #     # Recalculate the new state of the robot after the action
            #     next_self_state = self.propagate(state.self_state, action)
            #     # Make a copy of this state, to calculate the linear prediction
            #     next_self_state_copy = copy.deepcopy(next_self_state)
            #     # Create the list of the new state.
            #     # The first element is the new state after action, and next states are the states after the linear prediction of the current action. 
            #     next_poit_list = [[next_self_state_copy.px, next_self_state_copy.py, next_self_state_copy.vx, next_self_state_copy.vy]]
            #     # How many elements we need to predict
            #     size_difference = freq_time_param - tracklet.shape[0] - 1
            #     # In the case of non-holonomic movements we need to change r velocity 
            #     if self.kinematics == 'holonomic':
            #         copy_action = ActionXY(action.vx, action.vy)
            #     else: copy_action = ActionRot(action.v, 0)
            #     # Make a linear prediction of the robot positions using the current action
            #     for i in range(size_difference):
            #         next_self_state_copy = self.propagate(next_self_state_copy, copy_action)
            #         next_poit_list.append([next_self_state_copy.px, next_self_state_copy.py, next_self_state_copy.vx, next_self_state_copy.vy])
            #     # Concatenate the previous robot positions, the future position from the action and the predicted positions
            #     next_poit_array = np.array(next_poit_list)
            #     new_tracklet = np.concatenate((tracklet, next_poit_array), axis=0)
            #     new_tracklet = np.expand_dims(new_tracklet, axis=0)
            #     # Calculate is the trajectory will be social
            #     y_pred = classificator.test(new_tracklet)
            #     pred_list.append(y_pred.numpy()[0][0])

            #     # Calculate the next states of the humans and the reward for the action
            #     if self.query_env:
            #         # The choosing between two reward functions
            #         if dist_real is None: next_human_states, reward, _, _ = self.env.onestep_lookahead(action)
            #         else: next_human_states, reward, _, _ = self.env.onestep_lookahead(action, [dist_best, dist_real])
            #     else:
            #         next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
            #                         for human_state in state.human_states]
            #         # The choosing between two reward functions
            #         if dist_real is None: reward = self.compute_reward(next_self_state, next_human_states)
            #         else: reward = self.compute_reward(next_self_state, next_human_states, [dist_best, dist_real, action])
                
            #     # Unify the robot state vector with humans state vector
            #     batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
            #                                 for next_human_state in next_human_states], dim=0)
            #     # The output of rotate in the agent-centric coordinate: dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum
            #     rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
            #     if self.with_om:
            #         if occupancy_maps is None:
            #             occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
            #         rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
            #     # VALUE UPDATE
            #     next_state_value = self.model(rotated_batch_input).data.item()
            #     value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
            #     # Cummulate sum
            #     value = value + self.soc_coef * y_pred.numpy()[0][0]
            #     self.action_values.append(value)
            #     if value > max_value:
            #         max_value = value
            #         max_action = action
            #         y_pred_best = y_pred.numpy()[0][0]
            # # if len(set(pred_list)) == 2 and y_pred_best == 0.0:
            # #     doublelist = sorted(zip(self.action_values, pred_list))
            # #     for i in doublelist:
            # #         print(i)
            # #     print("___")
            # #     print(max_value, y_pred_best)
            # #     print()
            # # print(np.mean(pred_list))
            # # print(y_pred_best)
            # # print(pred_list)
            if max_action is None:
                raise ValueError('Value network is not well trained. ')
        if self.phase == 'train':
            self.last_state = self.transform(state)
        return max_action
