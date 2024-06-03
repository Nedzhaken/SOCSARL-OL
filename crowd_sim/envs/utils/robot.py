from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionRot
import math
# import pprint module
from pprint import pprint


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.next_goal_list = []
        self.classificator = None
        self.env = None
        self.freq_time_param = 16
        self.soc_class_coef = 1
        self.dist_treshold = 2.0
        self.pred_policy = ORCA()
        # set safety space for ORCA in non-cooperative simulation
        if self.visible:
            self.pred_policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            self.pred_policy.safety_space = 0

    def add_goal(self, goal):
        self.next_goal_list.append(goal)

    def set_env(self, env):
        self.env = env

    def get_next_goal(self):
        if self.next_goal_list:
            return self.next_goal_list[0]
        else:
            return None
        
    def remove_goal(self):
        self.next_goal_list.pop(0)

    def clean_goals(self):
        self.next_goal_list = []

    #ORCA part nav
    # def act(self, ob, dist_best = None, dist_real = None):
    #     if self.policy is None:
    #         raise AttributeError('Policy attribute has to be set!')
    #     state = JointState(self.get_full_state(), ob)
    #     dist = self.dist_to_closest_human(ob)
    #     if type(self.policy) == ORCA:
    #         action = self.policy.predict(state)
    #     elif dist < self.dist_treshold:
    #         action = self.policy.predict(state, dist_best, dist_real)
    #     else:
    #         action = self.pred_policy.predict(state)
    #         if self.kinematics != 'holonomic': action = self.actionXY_to_actionROT(action)
    #     return action
        
    # def act_test(self, ob, tracklet, classificator, dist_best = None, dist_real = None):
    #     if self.policy is None:
    #         raise AttributeError('Policy attribute has to be set!')
    #     state = JointState(self.get_full_state(), ob)
    #     dist = self.dist_to_closest_human(ob)
    #     if dist < self.dist_treshold:
    #         action = self.policy.predict_class(state, classificator, tracklet, self.freq_time_param, dist_best, dist_real)
    #     else:
    #         action = self.pred_policy.predict(state)
    #         if self.kinematics != 'holonomic': action = self.actionXY_to_actionROT(action)
    #     return action
    
    #Normal
    def act(self, ob, dist_best = None, dist_real = None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        if type(self.policy) == ORCA:
            action = self.policy.predict(state)
        else:
            action = self.policy.predict(state, dist_best, dist_real)
        return action
    
    def act_test(self, ob, tracklet, classificator, dist_best = None, dist_real = None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict_class(state, classificator, tracklet, self.freq_time_param, dist_best, dist_real)
        return action
    
    def dist_to_closest_human(self, ob):
        dist_min = 1000
        for human in ob:
            dist = ((self.px - human.px)**2 + (self.py - human.py)**2)**0.5 - human.radius - self.radius
            if dist < dist_min:
                dist_min = dist
        return dist_min
    
    def actionXY_to_actionROT(self, action):
        alpha = math.atan2(action.vy, action.vx)
        v = action.vx / math.cos(alpha)
        r = alpha - self.theta
        action = ActionRot(v, r)
        return action

