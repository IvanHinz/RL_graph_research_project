# Robustness of RL Agents in graph-based environment

This repo contains the code for my research project on how to make RL agents perform when the underlying graph changes.


## Repo layout
```text
src/
├── adversarial_attacks.py    # EACN, EAAN, FGSM variants
├── analyze_robustness.py     # evaluation of agents' performance in altered graph-env
├── eval_avg_returns.py       # plot avg-return training curves 
├── evaluate.py               # evaluation on changed graph 
├── graph_env.py              # graph environment build with NetworkX
├── models.py                 # A2C and DQN implementations                
├── real_graph_data.py        # OSMnx helpers for street networks
├── train.py                  # main training script 
└── utils.py                  # moving-average utilities