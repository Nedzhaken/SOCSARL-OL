import sys
import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from simul_classificator import TrackletsClassificator
from crowd_sim.envs.utils.action import ActionRot


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', default=False, action='store_true')
    # parser.add_argument('--video_file', type=str, default=None)
    # parser.add_argument('--video_file', type=str, default='animation10.gif')
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # create test log file
    if os.path.exists(os.path.join(args.model_dir, 'output_test.log')):
        key = input('Output_test.log already exists! Overwrite the file? (y/n)')
        if key == 'y' and not args.resume:
            os.remove(os.path.join(args.model_dir, 'output_test.log')) 
    log_file = os.path.join(args.model_dir, 'output_test.log')

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
    #                     datefmt="%Y-%m-%d %H:%M:%S")
    # configure logging and device    
    mode = 'a' if args.resume else 'w' 
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))
        torch.save(policy.get_model().state_dict(), 'rl_model.pt', _use_new_zipfile_serialization=False)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    robot.set_env(env)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.pred_policy.time_step = env.time_step
    
    ol = False
    if (policy.name == 'SARL-SOC' or ol):
        # __________________________________________________________________________
        robot.freq_time_param = 16
        robot.soc_class_coef = 0.75
        policy.movement_predictor = ORCA()
        policy.movement_predictor.time_step = env.time_step
        logging.info('The tracklet coef: %f', robot.soc_class_coef)

        model_name = 'model_4hz.pth'
        # model_name = '/home/iaroslav/Crowd_nav_ol/crowd_nav/model_4hz.pth'
        robot.classificator = TrackletsClassificator(hidden_size = robot.freq_time_param, model_name = model_name, device = device)
        # __________________________________________________________________________

    if args.visualize:
        if (policy.name == 'SARL-SOC' or ol):
            explorer.run_k_episodes(1, args.phase, print_failure=True, from_test = True, ol = ol, start_episode = args.test_case)
        else:
            explorer.run_k_episodes(1, args.phase, print_failure=True, from_test = True, start_episode = args.test_case)

        name = None
        if args.traj:
            name = policy.name + '_' + env.movement_pattern + '_' + str(args.test_case) + '.png'
            env.render('traj', name)
        else:
            if args.video_file:
                if ol: name = policy.name + '_OL_' + env.movement_pattern + '_' + str(args.test_case) + '.gif'
                else: name = policy.name + '_' + env.movement_pattern + '_' + str(args.test_case) + '.gif'
            env.render('video', name)

    else:
        # if (policy.name == 'SARL-SOC' or ol):
        #     explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, from_test = True, ol = ol)
        # else:
        #     explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, from_test = True)
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, from_test = True, ol = ol)

    # ol = True
    # if (policy.name == 'SARL-SOC'):
    #     # __________________________________________________________________________
    #     robot.freq_time_param = 16
    #     robot.soc_class_coef = 0.75
    #     logging.info('The tracklet coef: %f', robot.soc_class_coef)

    #     model_name = 'model_4hz.pth'
    #     # model_name = '/home/iaroslav/Crowd_nav_ol/crowd_nav/model_4hz.pth'
    #     robot.classificator = TrackletsClassificator(hidden_size = robot.freq_time_param, model_name = model_name, device = device)
    #     # The constants for online learning 
    #     if ol:
    #         logging.info('Online learning module is activated')
    #         # We need to understand the length of input tracklets for the model. The second element of modules() is the first layer
    #         first_layers = [module for module in robot.classificator.model.modules()][1]
    #         # We can take hidden_size only from GRU layer
    #         if type(first_layers) == nn.GRU:
    #             tracklet_len = first_layers.hidden_size
    #         else: tracklet_len = 20
    #         check_update = 3
    #         coef_update = 0.5
    #         soc_extra_dist_th = 0.9
    #         # The variables for online learning 
    #         robot_tracklet_list = []
    #         robot_tracklet_label_list = []
    #         human_tracklet_list = []
    #         counter_new_tracklet = 0
    #         # Copy the model to have a basic model for each new start of simulation
    #         model_for_train = copy.deepcopy(robot.classificator)
    #     # __________________________________________________________________________            

    # if args.visualize:
    #     ob = env.reset(args.phase, args.test_case)
    #     done = False
    #     last_pos = np.array(robot.get_position())
    #     if env.movement_pattern == 'center':
    #         goals = [[0.0, -env.circle_radius],
    #                 [0.0, 0.0],
    #                 [env.circle_radius, 0.0],
    #                 [-env.circle_radius, 0.0],
    #                 [0.0, 0.0],
    #                 [0.0, -env.circle_radius],
    #                 [0.0, 0.0],
    #                 [0.0, env.circle_radius],
    #                 [0.0, 0.0],
    #                 [env.circle_radius, 0.0],
    #                 [-env.circle_radius, 0.0],
    #                 [0.0, 0.0]]
    #         # goals = [[0.0, -env.circle_radius],
    #         #             [0.0, 0.0],
    #         #             [env.circle_radius, 0.0],
    #         #             [-env.circle_radius, 0.0],
    #         #             [0.0, 0.0],
    #         #             [0.0, -env.circle_radius]]
    #         # goals = []
    #     else:
    #         goals = []
    #         px, py = robot.get_position()
    #         gx, gy = env.next_circle_goal(px, py)
    #         robot.gx, robot.gy = gx, gy
    #         number_goals = int(env.circle_number * 360 / abs(env.step_degree))
    #         for _ in range(number_goals - 1):
    #             n_px, n_py = gx, gy
    #             gx, gy = env.next_circle_goal(n_px, n_py, step_def = env.step_degree)
    #             goals.append([gx, gy])
            
    #     # robot.add_goal(list(last_pos))
    #     # goals = [[0.0, -4.0], [0.0, 0.0], [4.0, 0.0], [-4.0, 0.0], [0.0, 0.0]]
    #     # 10 degrees, 4 meters, 2 circle
    #     # goals = [[1.368080573302676, -3.7587704831436333], [2.0000000000000004, -3.4641016151377544], [2.571150438746157, -3.0641777724759125], [3.064177772475911, -2.5711504387461583], [3.4641016151377535, -2.0000000000000018], [3.7587704831436337, -1.3680805733026744], [3.939231012048832, -0.6945927106677215], [4.0, -9.797174393178826e-16], [3.9392310120488325, 0.6945927106677197], [3.758770483143634, 1.3680805733026733], [3.4641016151377553, 1.9999999999999987], [3.064177772475913, 2.571150438746156], [2.5711504387461583, 3.064177772475911], [2.0000000000000013, 3.464101615137754], [1.3680805733026762, 3.7587704831436333], [0.6945927106677225, 3.939231012048832], [1.133107779529596e-15, 4.0], [-0.6945927106677203, 3.9392310120488325], [-1.368080573302674, 3.7587704831436337], [-1.9999999999999991, 3.464101615137755], [-2.5711504387461575, 3.064177772475912], [-3.0641777724759116, 2.571150438746158], [-3.464101615137755, 1.9999999999999998], [-3.7587704831436333, 1.3680805733026755], [-3.939231012048832, 0.6945927106677211], [-4.0, 4.898587196589413e-16], [-3.939231012048832, -0.6945927106677219], [-3.7587704831436337, -1.3680805733026746], [-3.464101615137756, -1.9999999999999976], [-3.0641777724759147, -2.5711504387461543], [-2.571150438746161, -3.0641777724759094], [-2.000000000000005, -3.4641016151377517], [-1.3680805733026842, -3.75877048314363], [-0.6945927106677283, -3.939231012048831], [-7.840215437089414e-15, -4.0], [0.6945927106677129, -3.939231012048834], [1.368080573302666, -3.758770483143637], [1.9999999999999942, -3.464101615137758], [2.5711504387461517, -3.064177772475917], [3.0641777724759067, -2.5711504387461637], [3.46410161513775, -2.000000000000008], [3.75877048314363, -1.3680805733026844], [3.9392310120488303, -0.6945927106677321], [4.0, -1.1637858475719386e-14], [3.9392310120488343, 0.6945927106677091], [3.7587704831436377, 1.368080573302663], [3.4641016151377606, 1.9999999999999893], [3.06417777247592, 2.571150438746148], [2.5711504387461668, 3.0641777724759045], [2.00000000000001, 3.4641016151377486], [1.3680805733026862, 3.7587704831436293], [0.6945927106677331, 3.9392310120488303], [1.1791248815931098e-14, 4.0], [-0.6945927106677099, 3.9392310120488343], [-1.368080573302664, 3.7587704831436377], [-1.99999999999999, 3.46410161513776], [-2.5711504387461477, 3.06417777247592], [-3.064177772475905, 2.571150438746166], [-3.4641016151377486, 2.0000000000000107], [-3.7587704831436297, 1.3680805733026855], [-3.9392310120488303, 0.6945927106677315], [-4.0, 9.371642916660193e-15], [-3.939231012048834, -0.6945927106677113], [-3.7587704831436377, -1.3680805733026629], [-3.4641016151377615, -1.9999999999999882], [-3.0641777724759214, -2.5711504387461463], [-2.571150438746169, -3.0641777724759023], [-2.0000000000000138, -3.4641016151377464], [-1.3680805733026908, -3.758770483143628], [-0.6945927106677388, -3.939231012048829], [-1.4945642794690415e-14, -4.0]]
    #     # 10 degrees
    #     # robot.gx = 2.736161146605352
    #     # robot.gy = -7.5175409662872665
    #     # goals = [[2.571150438746157, -3.0641777724759125], [3.064177772475911, -2.5711504387461583], [3.4641016151377535, -2.0000000000000018], [3.7587704831436337, -1.3680805733026744], [3.939231012048832, -0.6945927106677215], [4.0, -9.797174393178826e-16], [3.9392310120488325, 0.6945927106677197], [3.758770483143634, 1.3680805733026733], [3.4641016151377553, 1.9999999999999987], [3.064177772475913, 2.571150438746156], [2.5711504387461583, 3.064177772475911], [2.0000000000000013, 3.464101615137754], [1.3680805733026762, 3.7587704831436333], [0.6945927106677225, 3.939231012048832], [1.133107779529596e-15, 4.0], [-0.6945927106677203, 3.9392310120488325], [-1.368080573302674, 3.7587704831436337], [-1.9999999999999991, 3.464101615137755], [-2.5711504387461575, 3.064177772475912], [-3.0641777724759116, 2.571150438746158], [-3.464101615137755, 1.9999999999999998], [-3.7587704831436333, 1.3680805733026755], [-3.939231012048832, 0.6945927106677211], [-4.0, 4.898587196589413e-16], [-3.939231012048832, -0.6945927106677219], [-3.7587704831436337, -1.3680805733026746], [-3.464101615137756, -1.9999999999999976], [-3.0641777724759147, -2.5711504387461543], [-2.571150438746161, -3.0641777724759094], [-2.000000000000005, -3.4641016151377517], [-1.3680805733026842, -3.75877048314363], [-0.6945927106677283, -3.939231012048831], [-7.840215437089414e-15, -4.0], [0.6945927106677129, -3.939231012048834]]
    #     # robot.px = 0
    #     # robot.py = -8.0
    #     # goals = [[2.736161146605352, -7.5175409662872665], [4.000000000000001, -6.928203230275509], [5.142300877492314, -6.128355544951825], [6.128355544951822, -5.142300877492317], [6.928203230275507, -4.0000000000000036], [7.517540966287267, -2.736161146605349], [7.878462024097664, -1.389185421335443], [8.0, -1.959434878635765e-15], [7.878462024097665, 1.3891854213354393], [7.517540966287268, 2.7361611466053466], [6.9282032302755105, 3.9999999999999973], [6.128355544951826, 5.142300877492312], [5.142300877492317, 6.128355544951822], [4.000000000000003, 6.928203230275508], [2.7361611466053524, 7.5175409662872665], [1.389185421335445, 7.878462024097664], [2.266215559059192e-15, 8.0], [-1.3891854213354407, 7.878462024097665], [-2.736161146605348, 7.517540966287267], [-3.9999999999999982, 6.92820323027551], [-5.142300877492315, 6.128355544951824], [-6.128355544951823, 5.142300877492316], [-6.92820323027551, 3.9999999999999996], [-7.5175409662872665, 2.736161146605351], [-7.878462024097664, 1.3891854213354422], [-8.0, 9.797174393178826e-16], [-7.878462024097664, -1.3891854213354438], [-7.517540966287267, -2.7361611466053493], [-6.928203230275512, -3.999999999999995], [-6.128355544951829, -5.142300877492309], [-5.142300877492322, -6.128355544951819], [-4.00000000000001, -6.928203230275503], [-2.7361611466053684, -7.51754096628726], [-1.3891854213354566, -7.878462024097662], [-1.568043087417883e-14, -8.0]]
    #     # 5 degrees
    #     # goals = [[0.6945927106677199, -3.9392310120488325], [1.0352761804100812, -3.8637033051562737], [1.368080573302676, -3.7587704831436333], [1.6904730469627984, -3.6252311481465997], [2.0000000000000004, -3.4641016151377544], [2.294305745404184, -3.276608177155967], [2.571150438746157, -3.0641777724759125], [2.8284271247461894, -2.8284271247461907], [3.064177772475911, -2.5711504387461583], [3.2766081771559663, -2.294305745404186], [3.4641016151377535, -2.0000000000000018], [3.6252311481466, -1.6904730469627969], [3.7587704831436337, -1.3680805733026744], [3.8637033051562732, -1.0352761804100827], [3.939231012048832, -0.6945927106677215], [3.984778792366982, -0.3486229709906333], [4.0, -9.797174393178826e-16], [3.984778792366982, 0.34862297099063133], [3.9392310120488325, 0.69459271066772], [3.8637033051562737, 1.0352761804100816], [3.758770483143634, 1.3680805733026735], [3.6252311481466006, 1.6904730469627962], [3.4641016151377553, 1.9999999999999987], [3.276608177155968, 2.2943057454041833], [3.064177772475913, 2.571150438746156], [2.828427124746191, 2.828427124746189], [2.5711504387461583, 3.064177772475911], [2.2943057454041855, 3.2766081771559663], [2.0000000000000013, 3.464101615137754], [1.6904730469627984, 3.6252311481465993], [1.3680805733026762, 3.7587704831436333], [1.0352761804100838, 3.863703305156273], [0.6945927106677225, 3.939231012048832], [0.34862297099063344, 3.984778792366982], [1.133107779529596e-15, 4.0], [-0.34862297099063116, 3.984778792366982], [-0.6945927106677203, 3.9392310120488325], [-1.0352761804100816, 3.8637033051562737], [-1.368080573302674, 3.7587704831436337], [-1.6904730469627973, 3.6252311481466], [-1.9999999999999991, 3.464101615137755], [-2.2943057454041833, 3.276608177155968], [-2.5711504387461575, 3.064177772475912], [-2.82842712474619, 2.8284271247461903], [-3.0641777724759116, 2.571150438746158], [-3.2766081771559676, 2.2943057454041837], [-3.464101615137755, 1.9999999999999998], [-3.6252311481465997, 1.690473046962798], [-3.7587704831436333, 1.3680805733026755], [-3.863703305156273, 1.035276180410084], [-3.939231012048832, 0.6945927106677211], [-3.984778792366982, 0.3486229709906328], [-4.0, 4.898587196589413e-16], [-3.984778792366982, -0.3486229709906318], [-3.939231012048833, -0.6945927106677183], [-3.863703305156274, -1.0352761804100796], [-3.758770483143635, -1.3680805733026713], [-3.6252311481466024, -1.6904730469627922], [-3.464101615137759, -1.999999999999993], [-3.276608177155973, -2.2943057454041758], [-3.06417777247592, -2.5711504387461477], [-2.828427124746199, -2.828427124746181], [-2.5711504387461663, -3.0641777724759045], [-2.2943057454041944, -3.27660817715596], [-2.0000000000000107, -3.464101615137748], [-1.6904730469628095, -3.6252311481465944], [-1.3680805733026875, -3.758770483143629], [-1.0352761804100963, -3.8637033051562697], [-0.6945927106677388, -3.939231012048829], [-0.34862297099064715, -3.984778792366981], [-1.4945642794690415e-14, -4.0], [0.3486229709906174, -3.9847787923669835]]
    #     # 90 degrees
    #     # robot.gx = 4.0
    #     # robot.gy = 0.0
    #     # 1 circle
    #     # goals = [[1.2246467991473533e-15, 4.0], [-4.0, 4.898587196589413e-16], [-7.347880794884119e-16, -4.0]]
    #     # 2 circle
    #     # goals = [[1.2246467991473533e-15, 4.0], [-4.0, 4.898587196589413e-16], [-7.347880794884119e-16, -4.0], [4.0, -9.797174393178826e-16],
    #     #          [1.2246467991473533e-15, 4.0], [-4.0, 4.898587196589413e-16], [-7.347880794884119e-16, -4.0]]
    #     # robot.px = 0
    #     # robot.py = -6.0
        
    #     for goal in goals:
    #         robot.add_goal(goal)
    #     freq_time_mode = int(robot.freq_time_param * robot.soc_class_coef)
    #     # Calculate the ideal distance for the robot way
    #     px, py, gx, gy = robot.px, robot.py, robot.gx, robot.gy
    #     dist_best = ((px - gx)**2 + (py - gy)**2)**0.5
    #     for j in robot.next_goal_list:
    #         px, py, gx, gy = gx, gy, j[0], j[1]
    #         dist_best += ((px - gx)**2 + (py - gy)**2)**0.5
    #     # Calculate the ideal time for the robot way
    #     time_best = dist_best/robot.v_pref
    #     # The variables which represente the real distance and time of the robot way
    #     time_real = 0
    #     dist_real = 0
    #     # The list with robot goal positions
    #     past_goal_list = []
    #     while not done:
    #         # We want to have a list of the robot goal positions to calculate some metrics later 
    #         past_goal_list.append((robot.gx, robot.gy))
    #         if (policy.name == 'SARL-SOC'):
    #             if len(env.states[-freq_time_mode + 2:]) > freq_time_mode - 3:
    #                 pos = np.array([env.states[-i-1][0].position for i in reversed(range(len(env.states[-freq_time_mode + 2:])))])
    #                 vel = np.array([env.states[-i-1][0].velocity for i in reversed(range(len(env.states[-freq_time_mode + 2:])))])
    #                 tracklet = np.concatenate([pos, vel], 1)
    #                 rp = robot.get_position()
    #                 rv = robot.get_velocity()
    #                 robot_array = np.array([[rp[0], rp[1], rv[0], rv[1]]])
    #                 tracklet = np.concatenate((tracklet, robot_array), 0)
    #                 action = robot.act_test(ob, tracklet, robot.classificator, dist_best, dist_real)
    #             else: action = robot.act(ob, dist_best, dist_real)               
    #         else: action = robot.act(ob, dist_best, dist_real)
    #         ob, _, done, info = env.step(action)
    #         if type(action)==ActionRot:
    #             # for nonholonomic
    #             dist_real = dist_real + action.v*robot.time_step
    #         else:
    #             # for holonomic
    #             dist_real = dist_real + ((action.vx**2+action.vy**2)**0.5)*robot.time_step
    #         current_pos = np.array(robot.get_position())
    #         logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
    #         last_pos = current_pos
    #         # ________________________________________
    #         # Online learning module
    #         if ol and policy.name == 'SARL-SOC':
    #             # After each step we try to find the full tracklet and if the number of full tracklets is enough we do the update of the weight
    #             if len(env.states) == tracklet_len * (counter_new_tracklet + 1):
    #                 # Take last new robot tracklet and goals of robot
    #                 pos = np.array([env.states[-j-1][0].position for j in reversed(range(len(env.states[-tracklet_len:])))])
    #                 vel = np.array([env.states[-j-1][0].velocity for j in reversed(range(len(env.states[-tracklet_len:])))])
    #                 goal = np.array(past_goal_list[-tracklet_len:])
    #                 tracklet = np.concatenate([pos, vel, goal], 1)
    #                 robot_tracklet_list.append(tracklet)
    #                 # Take a last new human tracklets                        
    #                 pos_h = np.array([env.states[-j-1][1] for j in reversed(range(len(env.states[-tracklet_len:])))])
    #                 # Create a list of lists. The len equals the number of people
    #                 human_list =  [[] for j in range(len(pos_h[0]))]
    #                 # humans is the list of FullStates of all human per one time step 
    #                 for humans in pos_h:
    #                     # human is one FullState, human_l is the list, which will save the position and velocity from human FullState 
    #                     for human, human_l in zip(humans, human_list):
    #                         pos = human.position
    #                         vel = human.velocity
    #                         goal = human.goal_position
    #                         human_l.append(pos + vel + goal)
    #                 # Save the human tracklets to retrain the model later
    #                 for j in human_list:
    #                     human_tracklet_list.append(j)
    #                 # Increase the counter to take the last positions for a new tracklet
    #                 counter_new_tracklet += 1

    #                 # If we have enough number of new tracklets we make a check for the model update
    #                 if int(len(robot_tracklet_list) % check_update) == 0:                                  

    #                     robot_data = np.array(robot_tracklet_list)
    #                     robot_tracklet_label_list = explorer.create_labels(robot_data, tracklet_length = tracklet_len, param = soc_extra_dist_th)
                            
    #                     # We need to use only last human tracklets
    #                     human_tracklet_df = np.asarray(human_tracklet_list[-check_update * len(human_list):])
    #                     # We need to cut the goal of humans for social classificator
    #                     if human_tracklet_df.shape[-1] != 4:
    #                         human_tracklet_df = np.delete(human_tracklet_df, [4, 5], -1)
    #                     y_pred = robot.classificator.test(human_tracklet_df)
    #                     print(y_pred.numpy()[0])
    #                     y_pred = model_for_train.test(human_tracklet_df)
    #                     print(y_pred.numpy()[0])
    #                     # If the last human movements are not enough social, it means that the model works bad and we need to retrain the model
    #                     if np.mean(y_pred.numpy()[0]) < coef_update:
    #                         print('Train')
    #                         # Prepare the human tracklets for retraining of the model
    #                         human_data = np.array(human_tracklet_list)

    #                         # All human tracklets are social
    #                         # human_label = np.ones(human_data.shape[0])

    #                         # Define the label of human tracklets by distance relation parameter
    #                         human_label = explorer.create_labels(human_data, tracklet_length = tracklet_len, param = soc_extra_dist_th)

    #                         # We need to cut the goal of humans for social classificator
    #                         if human_data.shape[-1] != 4:
    #                             human_data = np.delete(human_data, [4, 5], -1)

    #                         # The input for train
    #                         data_list = [human_label, human_data]

    #                         # Prepare the robot tracklets for retraining of the model
    #                         robot_data = []
    #                         robot_label = []
    #                         # We need to use for retrain only the tracklets, which are unsocial
    #                         for robot_tracklet_label, robot_tracklet in zip(robot_tracklet_label_list, robot_tracklet_list):
    #                             if robot_tracklet_label == 0:
    #                                 robot_label.append(robot_tracklet_label)
    #                                 robot_data.append(robot_tracklet)
    #                         # If we have unsocial robot tracklets we can add them to the train input
    #                         if robot_data:
    #                             robot_data = np.array(robot_data)
    #                             # Just drop the goal's columns if we have them
    #                             if robot_data.shape[-1] != 4:
    #                                 robot_data = np.delete(robot_data, [4, 5], -1)
    #                             robot_label = np.array(robot_label)
    #                             data_list[0] = np.concatenate((data_list[0], robot_label), axis = 0)
    #                             data_list[1] = np.concatenate((data_list[1], robot_data))
    #                         # train_data
    #                         print(len(data_list[0]))
    #                         model_for_train.train(data_list)
    #                         # Clean the list of human and robot tracklets so we would not retrain on the same data
    #                         robot_tracklet_list = []
    #                         robot_tracklet_label_list = []
    #                         human_tracklet_list = []
    #                     print()
    #         # ________________________________________
        
    #     logging.info('Finish: the extra distance relation is %.3f', dist_best/dist_real)
    #     name = None
    #     if args.traj:
    #         # env.render('traj', args.video_file)
    #         name = policy.name + '_' + env.movement_pattern + '_' + str(args.test_case) + '.png'
    #         env.render('traj', name)
    #     else:
    #         if args.video_file:
    #             name = policy.name + '_' + env.movement_pattern + '_' + str(args.test_case) + '.gif'
    #         env.render('video', name)
    #             # env.render('video', args.video_file)

    #     logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    #     if robot.visible and info == 'reach goal':
    #         human_times = env.get_human_times()
    #         logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    # else:
    #     if (policy.name == 'SARL-SOC'):
    #         explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, from_test = True, ol = ol)
    #     else:
    #         explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, from_test = True)


if __name__ == '__main__':
    main()
