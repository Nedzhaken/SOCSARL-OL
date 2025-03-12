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
    parser.add_argument('--ol', default=False, action='store_true')
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

    # configure logging and device    
    mode = 'a' if args.resume else 'w' 
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
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
    
    # configure online learning for SOCSARL
    ol = args.ol
    if (policy.name == 'SOCSARL'): logging.info('Online Learning: %s', ol)    

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
    if (policy.name == 'SOCSARL'):
        # __________________________________________________________________________
        robot.freq_time_param = 16
        robot.soc_class_coef = 0.75
        policy.movement_predictor = ORCA()
        policy.movement_predictor.time_step = env.time_step
        logging.info('The tracklet coef: %f', robot.soc_class_coef)

        model_name = 'social_module_4hz.pth'
        robot.classificator = TrackletsClassificator(hidden_size = robot.freq_time_param, model_name = model_name, device = device)
        # __________________________________________________________________________

    if args.visualize:
        if (policy.name == 'SOCSARL'):
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
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, from_test = True, ol = ol)

if __name__ == '__main__':
    main()
