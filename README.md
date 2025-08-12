## Reinforcement Learning Games Collection 
This repository contains implementations of four classic environments from Gymnasium trained using Policy Gradient methods in PyTorch.
Each game includes:

- A Jupyter Notebook (.ipynb) for interactive experimentation

- A Python script (.py) with argparse for command-line runs

- A requirements.txt listing dependencies

- Configurable training parameters

##  Environments Included
1.LunarLander-v3 – Land a spacecraft safely between flags

2.CartPole-v1 – Balance a pole on a moving cart

3.MountainCar-v0 – Drive up a hill despite limited power

4.Pong-v0 – Play Atari Pong using RL

##  Installation
Clone the repository and install dependencies:

``` bash

git clone https://github.com/your-username/RL-Games.git
cd RL-Games
pip install -r requirements.txt
```
##  Running the Scripts
Each environment has its own runnable .py file.

Example:

```bash

python LunarLander/main.py --episodes 100
```
Arguments:

Flag	Default	Description
--episodes	1	Number of episodes to train/run

##  Using the Notebooks
Open the corresponding .ipynb in Jupyter or VS Code for interactive training and visualization.

Example:

```bash

jupyter notebook LunarLander/LunarLander.ipynb
```
##  Features
- Implements vanilla policy gradient

- Configurable:

  - Reward-to-Go

  - Advantage Normalization

  - Batch Size

- Comparison plots for different configurations

##  Repository Structure
```

RL-Games/
│
├── LunarLander/
│   ├── LunarLander.ipynb
│   ├── main.py
│   ├── requirements.txt
│
├── CartPole/
│   ├── CartPole.ipynb
│   ├── main.py
│   ├── requirements.txt
│
├── MountainCar/
│   ├── MountainCar.ipynb
│   ├── main.py
│   ├── requirements.txt
│
├── Pong/
│   ├── Pong.ipynb
│   ├── main.py
│   ├── requirements.txt
│
└── README.md
```
## Acknowledgments

This Games was completed as part of the course Reinforcement learning(Ai3000) under the guidance of **Professor  Easwar Subramanian** at IIT Hyderabad. We thank them for their support and valuable feedback.

---
