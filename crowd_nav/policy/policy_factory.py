from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.socsarl import SOCSARL
from crowd_nav.policy.ST2 import ST2

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['socsarl'] = SOCSARL
policy_factory['st2'] = ST2