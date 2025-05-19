import torch
import networkx as nx

def evaluate_on_perturbed(model, graph, env_class, num_episodes=100, dqn_based_model=False):
    # create graph environment using "perturbed" graph
    env = env_class(graph)
    # adjacency tensor
    adj = torch.tensor(env.adjacency_matrix, dtype=torch.float32)
    num_nodes = env.num_nodes
    # shortest distance
    shortest = nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight')
    rewards_log = []
    optimal_misses = []
    first_right_rew_ep = num_episodes

    for episode in range(num_episodes):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        total_r = 0
        done = False

        while not done:
            valid = env.get_valid_actions()
            if dqn_based_model:
                with torch.no_grad():
                    output = model(state_tensor.unsqueeze(0))
            else:
                with torch.no_grad():
                    output = model(state_tensor.unsqueeze(0), adj, env.current_node)

            # check if DQN or A2C
            if isinstance(output, tuple):
                logits = output[0].squeeze(0)
                mask = torch.full((num_nodes,), float(-1e9))
                mask[valid] = 0.0
                masked_logits = logits + mask
                probs = torch.softmax(masked_logits, dim=-1)
                action = int(torch.argmax(probs))
            else:
                q_values = output.squeeze(0)
                q_masked = q_values.clone()
                invalid = list(set(range(num_nodes)) - set(valid))
                q_masked[invalid] = -float('inf')
                action = int(torch.argmax(q_masked))

            next_state, reward, done = env.step(action)
            state_tensor = torch.tensor(next_state, dtype=torch.float32)
            total_r += reward

        rewards_log.append(total_r)
        miss_opt = int(abs(total_r - shortest) > 1e-3)
        optimal_misses.append(miss_opt)
        if miss_opt == 0 and first_right_rew_ep == num_episodes:
            first_right_rew_ep = episode

    return rewards_log, optimal_misses, first_right_rew_ep