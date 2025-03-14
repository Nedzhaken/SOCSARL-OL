import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL


class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()

    def predict(self, *args):
        state = args[0]
        dist_best = None
        dist_real = None
        if len(args) == 3:
            dist_best = args[1]
            dist_real = args[2]

        """
        A base class for all methods that takes pairwise joint state as input to value network.
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
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None

            actions_tensor = None
            rewards_tensor = None
            for action in self.action_space:
                # Recalculate the new state of the robot after the action 
                next_self_state = self.propagate(state.self_state, action)
                # Calculate the next states of the humans and the reward for the action
                if self.query_env:
                    # The choosing between two reward functions
                    if dist_real is None: next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                    else: next_human_states, reward, done, info = self.env.onestep_lookahead(action, [dist_best, dist_real])
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
    
                if actions_tensor is None:
                    actions_tensor = rotated_batch_input
                    rewards_tensor = reward
                else:
                    actions_tensor = torch.cat((actions_tensor, rotated_batch_input))
                    rewards_tensor = torch.cat((rewards_tensor, reward))
            # VALUE UPDATE
            next_state_value = self.model(actions_tensor)
            values_tensor = rewards_tensor + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
            self.action_values = torch.reshape(values_tensor, (-1,)).tolist()
            max_action = self.action_space[torch.argmax(values_tensor).data.item()]

            if max_action is None:
                raise ValueError('Value network is not well trained. ')
        if self.phase == 'train':
            self.last_state = self.transform(state)
        return max_action

    def compute_reward(self, nav, humans, dist_list = None):
        # collision detection
        dmin = float('inf')
        collision = False
        for _, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            # The calculation of the distance relation to optimize the reward by pass distance
            if dist_list is None:
                reward = 1
            else:
                dist_best = dist_list[0]
                dist_real = dist_list[1]
                action = dist_list[2]
                if type(action)==ActionRot:
                    # for nonholonomic
                    dist_real = dist_real + action.v*self.env.time_step
                else:
                    # for holonomic
                    dist_real = dist_real + ((action.vx**2+action.vy**2)**0.5)*self.env.time_step
                reward = (dist_best/dist_real)
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5
        else:
            reward = 0
        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

