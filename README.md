# A Human-Oriented Cooperative Driving Approach Integrating Driving Intention, State, and Conflict

## 📌 Overview

This project implements **A Human-Oriented Cooperative Driving Approach Integrating Driving Intention, State, and Conflict** in CARLA, integrating:

- Frenet-based trajectory planning  
- Shared control between human driver and autonomous system  
- Reinforcement learning–compatible environment  
- Real-time simulation and visualization  

The system follows a modular pipeline:  Simulator → Planner → Controller → Execution → Visualization

## ⚙️ Installation

### 1. Install CARLA

Download CARLA from the official website: https://carla.org/

### 2. Install dependencies

```
# Clone the code to local machine
git clone https://github.com/i-Qin/HOCD
cd HOCD

# Create Conda environment
conda create -n HOCD python=3.8
conda activate HOCD

# Install dependencies
pip install -r requirements.txt
```
## 🚀 Usage

### 1. Run the CARLA server

- Ubuntu:

```bash
./CarlaUE4.sh
```

- Windows:
```bash
CarlaUE4.exe
```

### 2. Train RL model
```bash
python train_ppo.py
```

### 3. Run
The lateral controller type can be configured via the YAML file. Supported controller types include:

- MPC: Model Predictive Control
- Preview: Human-like preview controller
- G29: Real human control using a steering wheel
- SharedControl: Human-vehicle shared control framework

To enable real human driving:
- Set the controller type to G29
- Connect a Logitech G29 Driving Force Racing Wheel steering wheel
- The driver can manually switch driving intentions by using the paddle shifters on the steering wheel to toggle between different intentions.

```bash
python main.py
```



## 📚 Citation
```
@article{wang2026human,
  title={A Human-Oriented Cooperative Driving Approach: Integrating Driving Intention, State, and Conflict},
  author={Wang, Qin and Pang, Shanmin and Fang, Jianwu and Dong, Shengye and Liu, Fuhao and Xue, Jianru and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  publisher={IEEE}
}
```