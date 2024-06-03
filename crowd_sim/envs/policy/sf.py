import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class SF(Policy):
    def __init__(self):
        """
        

        """
        super().__init__()
        self.name = 'SF'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0

    def configure(self, config):
        # Desired force parameters
        self.relexation_time = config.getfloat('sf', 'relaxation_time', fallback = 0.5)
        self.goal_threshold = config.getfloat('sf', 'goal_threshold', fallback = 0.3)
        self.des_factor = config.getfloat('sf', 'des_factor', fallback = 1.0)
        # Social force parameters
        self.lambda_importance = config.getint('sf', 'lambda_importance', fallback = 2.0)
        self.gamma = config.getint('sf', 'gamma', fallback = 0.35)
        self.n = config.getint('sf', 'n', fallback = 2)
        self.n_prime = config.getint('sf', 'n_prime', fallback = 3)
        self.soc_factor = config.getfloat('sf', 'soc_factor', fallback = 7.1)
        # self.soc_factor = config.getfloat('sf', 'soc_factor', fallback = 10.1)
        return

    def set_phase(self, phase):
        return

    def predict(self, state, dist_best = None, dist_real = None):
        """
        """
        force = self.compute_forces(state)
        velocity = self.calculate_vel(state, force)
        action = ActionXY(velocity[0], velocity[1])

        return action
    
    def compute_forces(self, state):
        desired_force = self.get_desired_force(state.self_state)
        social_force = self.get_social_force(state)
        force = sum(desired_force, social_force)
        return force
    
    def calculate_vel(self, state, force):
        vel = np.array([[state.self_state.vx, state.self_state.vy]])
        max_speeds = np.array([state.self_state.v_pref])
        desired_velocity = vel + self.time_step * force
        desired_velocity = self.capped_velocity(desired_velocity, max_speeds)
        return(desired_velocity[0])

    def get_desired_force(self, self_state):
        """Calculates the force between this agent and the next assigned waypoint.
        If the waypoint has been reached, the next waypoint in the list will be
        selected.
        :return: the calculated force
        """
        pos = np.array([[self_state.px, self_state.py]])
        vel = np.array([[self_state.vx, self_state.vy]])
        goal = np.array([[self_state.gx, self_state.gy]])
        direction, dist = self.normalize(goal - pos)
        force = np.zeros((1, 2))
        max_speeds = np.array([[self_state.v_pref]])
        force[dist > self.goal_threshold] = (
            direction * max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > self.goal_threshold, :]
        force[dist <= self.goal_threshold] = -1.0 * vel[dist <= self.goal_threshold]
        force /= self.relexation_time
        return force * self.des_factor

    def get_social_force(self, state):
        """Calculates the social force between this agent and all the other agents
        belonging to the same scene.
        It iterates over all agents inside the scene, has therefore the complexity
        O(N^2). A better
        agent storing structure in Tscene would fix this. But for small (less than
        10000 agents) scenarios, this is just
        fine.
        :return:  nx2 ndarray the calculated force
        """
        state_list = [[state.self_state.px, state.self_state.py, state.self_state.vx, state.self_state.vy]]
        for human in state.human_states:
            state_list.append([human.px, human.py, human.vx, human.vy])
        state_array = np.array(state_list)
        pos = state_array[:, 0:2]
        vel = state_array[:, 2:4]
        pos_diff = self.each_diff(pos)  # n*(n-1)x2 other - self
        diff_direction, diff_length = self.normalize(pos_diff)
        vel_diff = -1.0 * self.each_diff(vel)  # n*(n-1)x2 self - other

        # compute interaction direction t_ij
        interaction_vec = self.lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = self.normalize(interaction_vec)
        # compute angle theta (between interaction and position difference vector)
        theta = self.vector_angles(interaction_direction) - self.vector_angles(
            diff_direction
        )
        # compute model parameter B = gamma * ||D||
        B = self.gamma * interaction_length

        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(self.n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(self.n * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * self.left_normal(
            interaction_direction
        )

        force = force_velocity + force_angle  # n*(n-1) x 2
        force = np.sum(force.reshape((state_array.shape[0], -1, 2)), axis=1)
        force = force[0]
        return force * self.soc_factor
    
    def normalize(self, vecs: np.ndarray):
        """Normalize nx2 array along the second axis
        input: [n,2] ndarray
        output: (normalized vectors, norm factors)
        """
        norm_factors = []
        for line in vecs:
            norm_factors.append(np.linalg.norm(line))
        norm_factors = np.array(norm_factors)
        normalized = vecs / np.expand_dims(norm_factors, -1)
        # get rid of nans
        for i in range(norm_factors.shape[0]):
            if norm_factors[i] == 0:
                normalized[i] = np.zeros(vecs.shape[1])
        return normalized, norm_factors
    
    def vec_diff(self, vecs: np.ndarray) -> np.ndarray:
        """r_ab
        r_ab := r_a − r_b.
        """
        diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
        return diff
    
    def each_diff(self, vecs: np.ndarray, keepdims=False) -> np.ndarray:
        """
        :param vecs: nx2 array
        :return: diff with diagonal elements removed
        """
        diff = self.vec_diff(vecs)
        # diff = diff[np.any(diff, axis=-1), :]  # get rid of zero vectors
        diff = diff[
            ~np.eye(diff.shape[0], dtype=bool), :
        ]  # get rif of diagonal elements in the diff matrix
        if keepdims:
            diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])

        return diff
    
    def vector_angles(self, vecs: np.ndarray) -> np.ndarray:
        """Calculate angles for an array of vectors
        :param vecs: nx2 ndarray
        :return: nx1 ndarray
        """
        ang = np.arctan2(vecs[:, 1], vecs[:, 0])  # atan2(y, x)
        return ang
    
    def left_normal(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
        return vecs
    
    def capped_velocity(self, desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)