# SOCSARL-OL

**[`Paper`](http://arxiv.org/abs/2406.11495) | [`Data`](http://thor.oru.se/magni.html) | [`Video`](https://youtu.be/bwmoqu_fyUo)**

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
The **Magni** folder contains the trajectory dataset used for training, which is located in the **data/Clean_data** folder. These trajectories consist of the human trajectories from the Magni dataset and their copies created using the ORCA algorithm to have non-social trajectories.
The **data/Tracklet_4s_4hz_v** folder contains preprocessed tracklets from the Magni dataset.  

- Each tracklet consists of **16 points** (`tracklet_points_number = 16`).
- The frequency of points in a tracklet is **4 Hz** (`tracklet_frequency_hz = 4`).
- Each point includes the agent's velocity.
  
## Files and Their Functions

#### `src/tracklets_creator.py`
- Contains the **TrackletsCreator** class, which processes the dataset by reading data, building trajectories, and splitting them into tracklets.
- **Key functions:**

  - #### `convert_trajectories_to_tracklets(tracklet_points_number, tracklet_frequency_hz, tracklet_csv_folder)`
    - Convert all robot and human trajectories to tracklets. Save tracklets as csv files.
    - **Parameters:**
      - `tracklet_points_number` – Number of points in each tracklet.
      - `tracklet_frequency_hz` – Sampling frequency (in Hz) of the points within each tracklet.
      - `tracklet_csv_folder` – Directory where the generated tracklet CSV files will be saved.

#### `src/tracklets_classificator.py`
- Contains the **TrackletsClassificator** class, which classifies tracklets as **social** or **non-social**.
- **Key functions:**

  - #### `train_and_test(train_data_gen, test_data_gen, criterion, optimizer, max_epochs, verbose = True)`
    - Trains, tests, and saves the social module. Additionally, it generates visualizations of loss and accuracy results.

  - #### `train_and_test_k_fold(dataset, criterion, max_epochs, k_folds = 5, verbose=True)`
    - Trains and tests the social module using **k-fold cross-validation**.

#### `src/plot.py`
- Contains the **Drawer** class, which draws the human trajectory and its robot non-social copy.
- **Key functions:**

  - #### `plot_pair_trajectoies(ped_traj, rob_traj, st, end, index, save)`
  - Plots the trajectories of a human (`ped_traj`) and a robot (`rob_traj`) within a specified time range and saves the visualization as a `.pdf` file.
  - **Parameters:**
    - `ped_traj` – List of human trajectories from the `.csv` file.
    - `rob_traj` – List of robot trajectories from the `.csv` file.
    - `st` – Start time for trajectory visualization.
    - `end` – End time for trajectory visualization.
    - `index` – Index of the human-robot trajectory pair to plot (`ped_traj[index]` and `rob_traj[index]`).
    - `save` – Boolean flag indicating whether to save the plot as a `.pdf` file.

## Setup of SOCSARL-RL (CrowdNav simulator and Magni package)
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install magni, crowd_sim, and crowd_nav into pip
```
pip3 install -e .
```
(optional) Uninstall magni, crowd_sim, and crowd_nav from pip
```
pip3 uninstall socsarl
```

## Getting Started in Magni
```
cd Magni
```
(Optional) To generate tracklets with different parameters (`tracklet_points_number, tracklet_frequency_hz`) from the saved trajectories, run the following commands:
```
python3 src/tracklets_creator.py
```
To train the social module, run the following commands:
```
python3 src/tracklets_classificator.py
```

## Getting Started in CrowdNav simulator
This repository is organized in two parts: **gym_crowd/** folder contains the simulation environment and
**crowd_nav/** folder contains codes for training and testing the policies. Details of the simulation framework can be found
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

## Robot Control Methods

The robot can be controlled using the following policies: **CADRL, LSTM-RL, ST2, SARL, SOCSARL,** and **SOCSARL-OL**.  

Pretrained models for **CADRL, LSTM-RL, ST2, and SARL** are available in the **Models** directory. To test these methods, move the corresponding **output** folder into **crowd_nav/data**.  

**SOCSARL and SOCSARL-OL** use the **SARL output** for implementation.  

### Example Test Commands:
```
python3 test.py --policy cadrl --model_dir data/output --phase test
python3 test.py --policy lstm_rl --model_dir data/output --phase test
python3 test.py --policy st2 --model_dir data/output --phase test
python3 test.py --policy sarl --model_dir data/output --phase test
python3 test.py --policy socsarl --model_dir data/output --phase test
python3 test.py --policy socsarl --ol --model_dir data/output --phase test
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
@article{okunevich2025ral,
  title = {Online Context Learning for Socially Compliant Navigation},
  author = {Iaroslav Okunevich and Alexandre Lombard and Tomas Krajnik and Yassine Ruichek and Zhi Yan},
  journal = {IEEE Robotics and Automation Letters},
  volume = {10},
  number = {5},
  pages={5042--5049},
  year={2025},
  publisher={IEEE}
}
```
