# SOCSARL-OL

**[`Paper`](http://arxiv.org/abs/2406.11495) | [`Data`]() | [`Video`](https://youtu.be/bwmoqu_fyUo)**

This repository contains the code and data for our paper titled **Online Context Learning for Socially-compliant Navigation**.

In the paper, we apply the CrowdNav simulator and the Thor-Magni dataset to train the social module.   
- [Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning, ICRA, 2019](https://github.com/vita-epfl/CrowdNav).
- [THOR-MAGNI: A Large-scale Indoor Motion Capture Recording of Human Movement and Robot Interaction, 2024](https://github.com/tmralmeida/magni-dash/tree/dash-public?tab=readme-ov-file).

## Abstract
Social robot navigation is a complex problem that requires the implementation of high-quality human-robot interactions to ensure that robot movements do not reduce human comfort or performance.
The objective of this research is to enhance the social efficiency and reliability of mobile robot navigation in a variety of context environments.
In order to achieve this, a new deep reinforcement learning method, SOCSARL-OL, has been proposed for robot navigation. The proposed method is designed to implement efficient human-robot interaction in different social contexts with the help of a social online learning module.
The efficacy of the proposed method was demonstrated in a variety of scenarios, with the most challenging scenario exhibiting an 8\% improvement in reaching the robot's goal without collision over the state-of-the-art methods.
The objective of this research is to enhance the social efficiency and reliability of mobile robot navigation in a variety of context environments.

## Method Overview
<img src="Conceptual_diagram.jpg" alt="Conceptual_diagram.jpg" width="1000" />

## Train the Social module on the Magni dataset
The Magni folder includes the dataset for the training, which is located in Clean_data folder.

read_Magni_dataset.py contains a class Simulator to read the Magni dataset and  animate the trajectories.

tracklets_creator.py contains a class TrackletsCreator, which is applied to read the data from the dataset, build the trajectories, split the trajectories into tracklets:

load_csv_from_folder(folder) - download a data from 'folder' with csv dataset files;

create_tracklets(step, hz, save = True, folder, velocity) - create the tracklets from the read trajectories and save these tracklets in .csv file in 'folder', if 'save' is True.

'step' is the number of points in one tracklet.

'hz' - is the frequency of points in a tracklet.

'velocity' - if this flag is True, tracklets will include also the agent's velocity.

```
cd Magni
```

## Setup of CrowdNav simulator
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip3 install -e .
```

## Getting Started in CrowdNav simulator
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python3 train.py --policy sarl
```
2. Test policies with 500 test cases.
```
python3 test.py --policy orca --phase test
python3 test.py --policy sarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python3 test.py --policy orca --phase test --visualize --test_case 0
python3 test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python3 test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python3 utils/plot.py data/output/output.log
```


## Simulation Clips
CADRL             | LSTM-RL
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/vrWsxPM.gif" width="400" />|<img src="https://i.imgur.com/6gjT0nG.gif" width="400" />
SARL             |  SOCSARL-OL
<img src="https://i.imgur.com/rUtAGVP.gif" width="400" />|<img src="https://i.imgur.com/UXhcvZL.gif" width="400" />

## Citation
If you are considering using this code, please reference the following:
```bibtex
@article{okunevich2024online,
  title={Online Context Learning for Socially-compliant Navigation},
  author={Okunevich, Iaroslav and Lombard, Alexandre and Krajnik, Tomas and Ruichek, Yassine and Yan, Zhi},
  journal={arXiv preprint arXiv:2406.11495},
  year={2024}
}
```
