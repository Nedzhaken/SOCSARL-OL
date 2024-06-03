from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.linear import Linear
import configparser


class Human(Agent):
    def __init__(self, config, section, env):
        super().__init__(config, section)
        if type(self.policy) != ORCA:
            policy_config_file = 'data/output/policy.config'
            policy_config = configparser.RawConfigParser()
            policy_config.read(policy_config_file)
            self.policy.configure(policy_config)
            # self.policy.set_phase('test')
            # self.policy.set_device('cpu')
            # self.policy.set_env(env)

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
