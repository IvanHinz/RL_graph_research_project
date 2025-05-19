import numpy as np
from scipy import stats
import networkx as nx
from graph_env import create_graph, GraphEnv, perturb_graph
from train import train_actor_critic
from src.adversarial_attacks import fgsm_actor, fgsm_critic, eaan_attack, eacn_attack
from evaluate import evaluate_on_perturbed
from real_graph_data import coordinates, create_osmnx_graph, cut_unnecessary_graph_edges

def evaluate_robustness(num_train_episodes,
                        num_eval_episodes,
                        epsilon,
                        adv_loss_weight,
                        entropy_beta,
                        real_graph_data=False,
                        num_graphs=100,
                        num_nodes=15,
                        num_edges=15):

    results = {
        'A2C': [], 'EAAN': [], 'EACN': [], 'FGSM_critic': [], 'FGSM_actor': [], 'A2C_DR': [], 'A2C_dist': [], 'EAAN_dist': [], 'EACN_dist': [], 'FGSM_critic_dist': [], 'FGSM_actor_dist': [], 'A2C_DR_dist': [],
        'DQN': [], 'FGSM': [], 'DQN_dist': [], 'FGSM_dist':[], 'DQN_DR':[], 'DQN_DR_dist':[]
    }

    # the dictionary which represents the number of first episode when the optimal dijkstra path was found
    first_episode = {'A2C': [], 'EAAN': [], 'EACN': [], 'FGSM_critic': [], 'FGSM_actor': [], 'A2C_DR': [], 'DQN': [], 'FGSM': [], 'DQN_DR':[]}
    first_episode_perturb = {'A2C': [], 'EAAN': [], 'EACN': [], 'FGSM_critic': [], 'FGSM_actor': [], 'A2C_DR': [], 'DQN': [], 'FGSM': [], 'DQN_DR':[]}
    optimal_on_perturbed = {'A2C': 0, 'EAAN': 0, 'EACN': 0, 'FGSM_critic': 0, 'FGSM_actor': 0, 'A2C_DR': 0, 'DQN': 0, 'FGSM': 0, 'DQN_DR':0}

    if real_graph_data:
        if num_graphs != len(coordinates):
            raise ValueError(
                f"Number of graphs ({num_graphs}) does not match the length of the list with the real data - coordinates ({len(coordinates)})"
            )

    for i in range(num_graphs):
        # create the graph and environment
        print(f"graph {i+1}/{num_graphs}")
        needed_graph_found = True
        while needed_graph_found:
            if real_graph_data:
                G = create_osmnx_graph(coordinates[i][0], coordinates[i][1], coordinates[i][2])
                G = cut_unnecessary_graph_edges(G, needed_edges=num_nodes)
            else:
                G = create_graph(num_nodes, num_edges)
            env = GraphEnv(G)
            G_perturbed = perturb_graph(G, num_changes=num_nodes//3) # num_nodes // 3
            # G_perturbed = perturb_graph_struct(G_perturbed, num_struct_changes=3)
            perturbed_opt_reward = nx.dijkstra_path_length(G_perturbed, 0, num_nodes-1, weight='weight')
            # filter if needed
            if perturbed_opt_reward > 90:
                print(f"Optimal reward is {perturbed_opt_reward}")
                needed_graph_found = False

        # train the agents
        # model_dqn, _, _, dqn_ep = train_dqn_masked(env, num_episodes=500, use_fgsm=False)
        # _ = env.reset()
        # model_fgsm, _, _, dqn_fgsm_ep = train_dqn_masked(env, num_episodes=500, use_fgsm=True)
        # _ = env.reset()
        model_a2c, _, _, a2c_ep = train_actor_critic(env, num_episodes=num_train_episodes, attack_fn=None, entropy_beta=entropy_beta)
        _ = env.reset()
        model_eaan, _, _, a2c_eaan_ep = train_actor_critic(env, num_episodes=num_train_episodes, attack_fn=eaan_attack, epsilon=epsilon, adv_loss_weight=adv_loss_weight, entropy_beta=entropy_beta)
        _ = env.reset()
        model_eacn, _, _, a2c_eacn_ep = train_actor_critic(env, num_episodes=num_train_episodes, attack_fn=eacn_attack, epsilon=epsilon, adv_loss_weight=adv_loss_weight, entropy_beta=entropy_beta)
        _ = env.reset()
        model_fgsm_critic, _, _, a2c_critic_ep = train_actor_critic(env, num_episodes=num_train_episodes, attack_fn=fgsm_critic, epsilon=epsilon, adv_loss_weight=adv_loss_weight, entropy_beta=entropy_beta)
        _ = env.reset()
        model_fgsm_actor, _, _, a2c_actor_ep = train_actor_critic(env, num_episodes=num_train_episodes, attack_fn=fgsm_actor, epsilon=epsilon, adv_loss_weight=adv_loss_weight, entropy_beta=entropy_beta)
        # model_dqn_dr, _, _, dqn_dr_ep = train_dqn_masked(env, num_episodes=500, use_fgsm=False, domain_randomization=True)
        _ = env.reset()
        model_a2c_dr, _, _, a2c_dr_ep = train_actor_critic(env, num_episodes=num_train_episodes, attack_fn=None, domain_randomization=True)

        # first_episode['DQN'].append(dqn_ep)
        # first_episode['FGSM'].append(dqn_fgsm_ep)
        first_episode['A2C'].append(a2c_ep)
        first_episode['EAAN'].append(a2c_eaan_ep)
        first_episode['EACN'].append(a2c_eacn_ep)
        first_episode['FGSM_critic'].append(a2c_critic_ep)
        first_episode['FGSM_actor'].append(a2c_actor_ep)
        # # first_episode['DQN_DR'].append(dqn_dr_ep)
        first_episode['A2C_DR'].append(a2c_dr_ep)

        # Dijkstra optimal path cost in perturbed graph
        opt_reward = nx.dijkstra_path_length(G_perturbed, 0, num_nodes-1, weight='weight')

        # evaluate the agents in the perturbed graph environment
        for label, model in zip(
            ['A2C','EAAN','EACN','FGSM_critic','FGSM_actor','A2C_DR'],
            # ['DQN', 'FGSM', 'DQN_DR'],
            [model_a2c, model_eaan, model_eacn, model_fgsm_critic, model_fgsm_actor, model_a2c_dr]
            # [model_dqn, model_fgsm, model_dqn_dr]
        ):
            rewards, _, found_ep = evaluate_on_perturbed(model, G_perturbed, GraphEnv, num_episodes=num_eval_episodes, dqn_based_model=False)
            first_episode_perturb[label].append(found_ep)
            avg_reward = max(rewards)
            print(f'Using {label} maximum reward was {avg_reward} and Dijkstra was {opt_reward}')
            results[label].append(avg_reward)
            results[f"{label}_dist"].append(abs(avg_reward - opt_reward))
            if abs(avg_reward - opt_reward) <= 0:
                optimal_on_perturbed[label] += 1

    print(f"On the training first episode when was founded an optimal path")
    means = {key: np.mean(value) if value else 0 for key, value in first_episode.items()}
    print(means)

    print(f"On the test first episode when was founded an optimal path")
    print(first_episode_perturb)
    means_perturb = {key: np.mean(value) if value else 0 for key, value in first_episode_perturb.items()}
    print(means_perturb)

    for key in optimal_on_perturbed:
        optimal_on_perturbed[key] /= num_graphs
    print(f'Optimal path found on perturbed in % of cases')
    print(optimal_on_perturbed)

    ci_rewards = {'DQN': 0, 'FGSM': 0, 'A2C':0, 'EAAN':0, 'EACN':0, 'FGSM_critic':0, 'FGSM_actor': 0,'DQN_DR': 0, 'A2C_DR': 0, }
    ci_avg_distance_to_dijkstra = {'DQN': 0, 'FGSM': 0, 'A2C':0, 'EAAN':0, 'EACN':0, 'FGSM_critic':0, 'FGSM_actor':0,'DQN_DR': 0, 'A2C_DR': 0}

    for label in ['A2C','EAAN','EACN','FGSM_critic','FGSM_actor','A2C_DR']:
        mean_r = np.mean(results[label])
        mean_d = np.mean(results[f"{label}_dist"])
        ci_r = stats.t.interval(0.95, len(results[label])-1, loc=mean_r, scale=stats.sem(results[label]))
        ci_d = stats.t.interval(0.95, len(results[f"{label}_dist"])-1, loc=mean_d, scale=stats.sem(results[f"{label}_dist"]))

        print(f"\n{label} avg reward: {mean_r} CI95: {ci_r}")
        print(f"{label} avg distance to dijkstra: {mean_d} CI95: {ci_d}")
        ci_rewards[label] = (mean_r, ci_r)
        ci_avg_distance_to_dijkstra[label] = (mean_d, ci_d)

    return ci_avg_distance_to_dijkstra, optimal_on_perturbed

if __name__ == "__main__":
    models = ['A2C','EAAN','EACN','FGSM_critic','FGSM_actor','A2C_DR']
    param_grid = [
        {
            "num_train_episodes": 500,
            "num_eval_episodes": 150,
            "epsilon": 1e6,
            "adv_loss_weight": 10,
            "entropy_beta": 0.01,
            "num_nodes": 30
        },
    ]

    # run models for different parameter configurations
    for idx, params in enumerate(param_grid):
        print(f"running experiment {idx + 1} with params={params}")
        # print(f"running experiment {idx + 1}")
        _, optimal_on_changed = evaluate_robustness(
            num_train_episodes=params["num_train_episodes"],
            num_eval_episodes=params["num_eval_episodes"],
            epsilon=params["epsilon"],
            adv_loss_weight=params["adv_loss_weight"],
            entropy_beta=params["entropy_beta"],
            real_graph_data=True,
            num_graphs=50,
            num_nodes=params["num_nodes"],
            num_edges=params["num_nodes"]
        )

        optimal_on_changed_diff = optimal_on_changed.copy()
        for model in models:
            optimal_on_changed_diff[model] = optimal_on_changed_diff[model] - optimal_on_changed['A2C']
        print(f"Difference between then simple model and adversarial attacks trained, params={params}")
        print(optimal_on_changed_diff)

