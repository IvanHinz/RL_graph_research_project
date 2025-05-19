import networkx as nx
import matplotlib.pyplot as plt
from graph_env import create_graph, GraphEnv
from train import train_actor_critic, train_dqn_masked
from src.adversarial_attacks import fgsm_actor, fgsm_critic, eaan_attack, eacn_attack
from utils import moving_average

def evaluate_plot_returns(n_nodes, n_edges):
    G = create_graph(n_nodes, n_edges)
    env = GraphEnv(G)

    optimal_cost = nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight')
    print("optimal cost according to dijkstra is :", optimal_cost)

    print("we train DQN without an attack")
    model_dqn, rewards_dqn, misses_dqn, _ = train_dqn_masked(env, num_episodes=550, use_fgsm=False)
    _ = env.reset()
    print("we train DQN attacked with FGSM")
    model_dqn_fgsm, rewards_fgsm, misses_fgsm, _ = train_dqn_masked(env, num_episodes=550, use_fgsm=True)
    _ = env.reset()
    print("we train A2C without an attack")
    model_a2c, rewards_a2c, misses_a2c, _ = train_actor_critic(env, num_episodes=550, attack_fn=None)
    _ = env.reset()
    print("we train A2C attacked with EAAN")
    model_eaan, rewards_eaan, misses_eaan, _ = train_actor_critic(env, num_episodes=550, attack_fn=eaan_attack)
    _ = env.reset()
    print("we train A2C attacked with EACN")
    model_eacn, rewards_eacn, misses_eacn, _ = train_actor_critic(env, num_episodes=550, attack_fn=eacn_attack)
    _ = env.reset()
    print("we train A2C attacked with FGSM on Critic")
    model_fgsm_critic, rewards_fgsm_critic, misses_fgsm_critic, _ = train_actor_critic(env, num_episodes=550, attack_fn=fgsm_critic)
    _ = env.reset()
    print("we train A2C attacked with FGSM on Actor")
    model_fgsm_actor, rewards_fgsm_actor, misses_fgsm_actor, _ = train_actor_critic(env, num_episodes=550, attack_fn=fgsm_actor)
    _ = env.reset()
    # print("we train DQN with domain randomization")
    # model_dqn_dr, rewards_dqn_dr, misses_dqn_dr, _ = train_dqn_masked(env, num_episodes=500, use_fgsm=False, domain_randomization=True)
    # print("we train A2C with domain randomization")
    # model_a2c_dr, rewards_a2c_dr, misses_a2c_dr, _ = train_actor_critic(env, num_episodes=500, attack=None, domain_randomization=True)

    # we do not look on rewards for environment_randomized because graph changes
    # training rewards
    plt.figure()
    plt.plot(moving_average(rewards_dqn, 50), label="DQN")
    plt.plot(moving_average(rewards_fgsm, 50), label="DQN_FGSM")
    plt.plot(moving_average(rewards_a2c, 50), label="A2C")
    plt.plot(moving_average(rewards_eaan, 50), label="A2C_EAAN")
    plt.plot(moving_average(rewards_eacn, 50), label="A2C_EACN")
    plt.plot(moving_average(rewards_fgsm_critic, 50), label="A2C_FGSM_Critic")
    plt.plot(moving_average(rewards_fgsm_actor, 50), label="A2C_FGSM_Actor")

    plt.axhline(optimal_cost, color='black', linestyle='--', label='optimal dijkstra')

    plt.title("mean_average_rewards, window_size = 20")
    plt.xlabel("episode")
    plt.ylabel("mean_average_reward")
    plt.legend()
    plt.savefig("rewards_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    num_nodes, num_edges = map(int, input().split())
    evaluate_plot_returns(num_nodes, num_edges)