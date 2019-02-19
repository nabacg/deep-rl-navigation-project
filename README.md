# Deep RL Navigation Project
This repository contains code demonstrating how to implement Deep Reinforcement Learning techniques to solve a Unity ML-Agents environment where a goal is to  train an agent to navigate a large world and collect yellow bananas, while avoiding blue bananas. 

Implementation using Python 3, PyTorch, Unity ML-Agents.

# Demo 

https://youtu.be/sITdR22gfPo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=sITdR22gfPo
" target="_blank"><img src="https://github.com/nabacg/deep-rl-navigation-project/blob/master/images/UnityEnvDemo.gif?raw=true" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

# Getting Started
Before running the code you'll need to install following python modules

## Requirements 

- [Python 3](https://www.python.org/) - currently only python 3.6 is supported due to Unity ML-Agents requirements
- [PyTorch](https://www.pytorch.org)
- [Numpy](http://www.numpy.org/)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) 
- [Jupyter Notebooks](https://jupyter.org/) - optional, needed to run [Report notebook](Report.ipynb) and [Training notebook](Training.ipynb) notebooks

## Installing requirements with Anaconda
Easiest way to install all dependencies is using [Anaconda](https://www.anaconda.com/distribution/). Install Anaconda for Python 3 using installer appropriate for your OS and once ready clone this repository and environment.yml file inside it.

```bash
git clone https://github.com/nabacg/deep-rl-navigation-project.git
cd deep-rl-navigation-project
conda env create -f environment.yml

```
It will take few minutes to install all packages. Once finished activate the newly created environment with

```bash
conda activate drl_nav
``` 

## Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
Then, place the file in the root folder in the Deep-RL-Navigation-Project GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

# Instructions 

## Project structure


- Report.ipynb - final report 
- Training.ipynb - demonstration on how to train agent from scratch, plus results
- src - python source files
 - dqn_agent.py contains Agent implementation and couple of helper functions
 - model.py - contains PyTorch Neural Network module for Q function approximation
 - replaybuffers.py - contain Experience Replay and Prioritized Experience Replay helper classes
 - main.py - delivers easy command line interface into other classes

## Jupyter notebooks
In order to train DQN Agent using Jupyter notebooks provided, start jupyter in project folder:

```bash
cd deep-rl-navigation-project
jupyter notebook 
``` 

then once Jupyter UI opens in your browser ([the default URL](http://localhost:8888/tree/) ),  open [Training notebook](Training.ipynb). 

If you'd rather view results or experiment with already trained agent open [Report notebook](Report.ipynb) instead.

## Command Line

It's also possible to train or test DQN Agent using command line only with help of main.py file. 

For example to train agent from scratch for 2000 episodes or until mean score of 13 is reached use this command:

```bash
python src/main.py --episodes 2000 --target_score 13.0
```

 To test already pretrained agent for 100 episodes using pretrained model from  qnetwork_model_weights.pth use: 
```bash

 python src/main.py --episodes 100 --mode test --input_weights qnetwork_model_weights.pth

```

The file exposes several command line arguments that allow to change various (hyper)parameters, those can be displayed using --help argument.

```bash
python src/main.py --help
usage: main.py [-h] [--env_file ENV_FILE] [--mode {train,test}]
               [--episodes EPISODES] [--eps_decay EPS_DECAY]
               [--eps_end EPS_END] [--target_score TARGET_SCORE]
               [--input_weights INPUT_WEIGHTS]
               [--output_weights OUTPUT_WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  --env_file ENV_FILE   Path to Unity Environment file, allows to change which
                        env is created. Defaults to Banana.app
  --mode {train,test}   Allows switch between training new DQN Agent or test
                        pretrained Agent by loading model weights from file
  --episodes EPISODES   Select how many episodes should training run for.
                        It should be set to a multiple of 100, otherwise target mean score
                        calculations will be meaningless.
  --eps_decay EPS_DECAY
                        Epsilon decay parameter value
  --eps_end EPS_END     Epsilon end value, after achieving which epsilon decay
                        stops
  --target_score TARGET_SCORE
                        Target traning score, when mean score over 100
                        episodes
  --input_weights INPUT_WEIGHTS
                        Path to Q Network model weights to load into DQN Agent
                        before training or testing
  --output_weights OUTPUT_WEIGHTS
                        Path to save Q Network model weights after training.

```
