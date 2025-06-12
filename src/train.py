import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import networkx as nx
import random
from src.models import DQN, A2C
from src.graph_env import perturb_graph, GraphEnv
from src.adversarial_attacks import fgsm_attack

def train_dqn_masked(env,
                     num_episodes=1000,
                     gamma=0.99,
                     use_fgsm=False,
                     domain_randomization=False,
                     train=True):
    num_nodes = env.num_nodes
    # online network
    online_net = DQN(num_nodes, hidden_dim=num_nodes)
    # target network
    target_net = DQN(num_nodes, hidden_dim=num_nodes)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
    # define the loss
    loss_fn = nn.MSELoss()
    buffer = deque(maxlen=10000)

    # start epsilon (used for probability of random action)
    eps = 1.0
    # minimal possible epsilon
    eps_min = 0.05
    # decay factor for epsilon
    eps_decay = 0.9995
    batch_size = 64
    # list to track total rewards per episode
    rewards_log = []
    # list to track episodes where the optimal path was missed
    optimal_misses = []
    # shortest path according to dijkstra
    shortest = nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight')
    # the first episode when the optimal according to dijkstra cost path was found
    founded_ep = num_episodes
    # regulates the frequency of environment changes in case of domain randomization
    domain_updated = num_episodes // 4
    # save initial_graph in case of domain randomization
    old_graph = env.our_graph.copy()
    for episode in range(num_episodes):
        # if needed we use domain_randomization
        if domain_randomization and train and episode % domain_updated == 0:
            # create the copy of the initial graph and perturb/change it
            g_pert = perturb_graph(old_graph, num_changes=num_nodes//3)
            env=GraphEnv(g_pert)
            shortest = nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight')
        # reset environment each new episode
        state = env.reset()
        # tensor of the state/observations
        state_tensor = torch.tensor(state, dtype=torch.float32)
        total_reward = 0
        done = False

        while not done:
            # if needed we attack the network using fast gradient sign method
            if use_fgsm:
                state_tensor = fgsm_attack(state_tensor.clone().detach(), online_net, epsilon=0.1)

            valid = env.get_valid_actions()
            # take random action or use the online_net
            if random.random() < eps:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    # q values for the current state and create a copy
                    q_values = online_net(state_tensor).squeeze(0)
                    q_values_masked = q_values.clone()
                    # list of the invalid actions
                    invalid = list(set(range(num_nodes)) - set(valid))
                    # set q_values of invalid action to the large negative value
                    q_values_masked[invalid] = -float(10**9)
                    # select action with the highest Q-value
                    action = int(torch.argmax(q_values_masked))

            # step to the next node according to the action
            next_state, reward, done = env.step(action)
            next_tensor = torch.tensor(next_state, dtype=torch.float32)
            # store transition to replay buffer
            buffer.append((state_tensor, action, reward, next_tensor, done))
            # update the current state
            state_tensor = next_tensor
            # sum to the episode reward the reward for the step
            total_reward += reward

            # if there are enough elements in buffer we learn
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                # state, action, reward, next_state, done
                st, act, rew, nst, d = zip(*batch)
                st = torch.stack(st)
                act = torch.tensor(act)
                rew = torch.tensor(rew, dtype=torch.float32)
                nst = torch.stack(nst)
                d = torch.tensor(d, dtype=torch.bool)
                # get q_values for the whole batch (and for all actions)
                q_values = online_net(st)
                # actions_tensor
                actions_tensor = act.unsqueeze(1)
                # choose needed q-values according to made actions
                q_for_actions = q_values.gather(dim=1, index=actions_tensor)
                q_predicted = q_for_actions.squeeze(1)
                # calculate q_values using target net
                with torch.no_grad():
                    q_next_all = target_net(nst)

                    for i in range(batch_size):
                        cur_node = nst[i].argmax().item()
                        # find valid actions
                        valid_next = [
                            j for j, w in enumerate(env.adjacency_matrix[cur_node]) if w != -1
                        ]
                        # get the list of invalid actions
                        invalid_next = list(set(range(num_nodes)) - set(valid_next))
                        # mask invalid actions
                        q_next_all[i][invalid_next] = -1e9

                    # q_next = target_net(nst).max(1)[0]
                    q_next = q_next_all.max(1)[0]
                    target_q = rew + gamma * q_next * (~d)
                loss = loss_fn(q_predicted, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # add to rewards list total_reward for the episode
        rewards_log.append(total_reward)
        # if 0 added to optimal_misses than the cost through episode was equal to dijkstra cost
        optimal_misses.append(int(abs(total_reward - shortest) > 1e-3))
        if optimal_misses[-1] == 0 and founded_ep==num_episodes:
            founded_ep = episode
        # reduce the probability of taking random action
        if eps > eps_min:
            eps *= eps_decay
        if (episode+1) % 50 == 0:
            if use_fgsm:
                print(f"DQN attacked with FGSM on episode {episode + 1}, reward={total_reward}, miss_optimal={optimal_misses[-1]}")
            else:
                print(f"DQN without an attack on episode {episode + 1}, reward={total_reward}, miss_optimal={optimal_misses[-1]}")
        # more frequently or less frequently make updates
        # lr e-2
        if (episode+1) % 100 == 0:
            target_net.load_state_dict(online_net.state_dict())

    return online_net, rewards_log, optimal_misses, founded_ep


def train_actor_critic(
        env,
        num_episodes=1000,
        gamma=0.99,
        attack_fn=None,
        epsilon=10**8,  # 8
        domain_randomization=False,
        train=True,
        entropy_beta=0.01,  # 0.01
        adv_loss_weight=10  # 10
    ):
    num_nodes = env.num_nodes
    # actor-critic model that uses simple GCN layer
    model = A2C(num_nodes, hidden_dim=env.num_nodes * 2) # 32 # env.num_nodes * 2
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # list to track total rewards per episode
    rewards_log = []
    # list to track episodes where the optimal path was missed
    optimal_misses = []
    # shortest path according to dijkstra
    shortest = nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight')
    # the first episode when the optimal according to dijkstra cost path was found
    founded_ep = num_episodes
    # regulates the frequency of environment changes in case of domain randomization
    domain_updated = num_episodes // 4
    # adjacency matrix copied for all environment
    adj_tensor = torch.tensor(env.adjacency_matrix, dtype=torch.float32)
    # save initial_graph in case of domain randomization
    old_graph = env.our_graph.copy()
    for episode in range(num_episodes):
        # if needed we use domain randomization
        if domain_randomization and train and episode % domain_updated == 0:
            # create the copy of the initial graph and perturb/change it
            g_pert = perturb_graph(old_graph, num_changes=num_nodes//3)
            env = GraphEnv(g_pert)
            shortest = nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight')
            adj_tensor = torch.tensor(env.adjacency_matrix, dtype=torch.float32)

        # reset environment each new episode
        state = env.reset()
        # tensor of the state/observations
        state_oh = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        states_clean = []
        actions = []
        values = []
        log_probs = []
        rewards_ep = []
        masks = []
        nodes = []
        valid_actions_list = []
        entropies = []

        while not done:
            # track the nodes sequence for later use
            nodes.append(env.current_node)
            # get logits and value from action critic
            logits, value = model(state_oh.unsqueeze(0), adj_tensor, env.current_node)
            logits = logits.squeeze(0)  # more to dimension [num_nodes]

            valid = env.get_valid_actions()
            # track the sequence of valid_actions
            valid_actions_list.append(valid)
            # mask for invalid actions
            mask = torch.full((num_nodes,), float(-1e9))
            mask[valid] = 0.0
            # apply the mask to logits making the probability going to the invalid node zero
            masked_logits = logits + mask
            # convert masked logits to probabilities
            probs = torch.softmax(masked_logits, dim=-1)
            # categorical distribution using probabilities because action space is discrete
            dist = torch.distributions.Categorical(probs)
            # save sequence of entropy of distibutions
            entropies.append(dist.entropy())
            # sample action from distribution
            action = dist.sample()

            # save the sequence of state/observation for later use
            states_clean.append(state_oh)
            # save the sequence of actions for later use
            actions.append(action)
            # save the sequence of critic values for later use
            values.append(value)
            # store the log_probability of chosen action (need for loss)
            log_probs.append(dist.log_prob(action))

            # step to the next node according to the action
            next_state, reward, done = env.step(action.item())
            # nodes.append(env.current_node)
            total_reward += reward

            # store immediate reward
            rewards_ep.append(torch.tensor([reward], dtype=torch.float32))
            masks.append(torch.tensor([1 - done], dtype=torch.float32))

            state_oh = torch.tensor(next_state, dtype=torch.float32)

        nodes.append(env.current_node)

        # add to rewards list total_reward for the episode
        rewards_log.append(total_reward)
        # if 0 added to optimal_misses than the cost through episode was equal to dijkstra cost
        optimal_misses.append(int(abs(total_reward - shortest) > 1e-3))

        if optimal_misses[-1] == 0 and founded_ep == num_episodes:
            # print(nodes)
            founded_ep = episode

        # compute and store discounted returns
        returns = []
        Gt = torch.tensor([0.0])
        for r, m in zip(reversed(rewards_ep), reversed(masks)):
            Gt = r + gamma * Gt * m
            returns.insert(0, Gt)
        returns = torch.cat(returns)  # (T,)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze(-1)
        # calculate advantage
        advantage_clean = returns - values

        # calculate mean entropy of probability distribution
        dist_entropy = torch.stack(entropies).mean()
        # actor loss
        actor_loss_clean = -(log_probs * advantage_clean.detach()).mean() - entropy_beta * dist_entropy
        # critic loss - mse of the advantage
        critic_loss_clean = advantage_clean.pow(2).mean()

        # applies adversarial attack if needed
        if attack_fn is not None:
            actor_loss_adv = []
            critic_loss_adv = []

            for i in range(len(states_clean)):
                s_clean = states_clean[i]
                a = actions[i]
                ret = returns[i]
                m = masks[i]
                cur_node = nodes[i]

                s_adv = attack_fn(s_clean.clone(), model, cur_node, epsilon)
                # s_adv = s_clean.clone() # for test
                logits_unused, value_unused = model(s_clean.unsqueeze(0), adj_tensor, cur_node)
                # print(f"Value simple is {value_unused}")
                # print(env.current_node)
                logits_adv, value_adv = model(s_adv.unsqueeze(0), adj_tensor, cur_node)
                # print(f"Value adversarial is {value_adv}")
                logits_adv = logits_adv.squeeze(0)
                # mask actions
                mask_adv = torch.full((num_nodes,), float(-1e9))
                valid_actions_adv = valid_actions_list[i]
                mask_adv[valid_actions_adv] = 0.0

                masked_logits_adv = logits_adv + mask_adv
                # compute policy distribution
                probs_adv = torch.softmax(masked_logits_adv, dim=-1)
                dist_adv = torch.distributions.Categorical(probs_adv)

                log_prob_adv = dist_adv.log_prob(a)
                advantage_adv = ret - value_adv

                actor_loss_i = -(log_prob_adv * advantage_adv.detach())
                critic_loss_i = advantage_adv.pow(2)

                actor_loss_adv.append(actor_loss_i)
                critic_loss_adv.append(critic_loss_i)

            actor_loss_adv = torch.stack(actor_loss_adv).mean()
            critic_loss_adv = torch.stack(critic_loss_adv).mean()

            # total loss final
            actor_loss_total = actor_loss_clean + adv_loss_weight * actor_loss_adv
            critic_loss_total = critic_loss_clean + adv_loss_weight * critic_loss_adv
        else:
            # if we have no adversarial attack during training
            actor_loss_total = actor_loss_clean
            critic_loss_total = critic_loss_clean

        loss = actor_loss_total + 0.5 * critic_loss_total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            if attack_fn is not None:
                print(
                    f"A2C attacked with {attack_fn.__name__} on episode {episode + 1}, reward={total_reward}, miss_optimal={optimal_misses[-1]}")
            else:
                print(f"A2C without an attack on episode {episode + 1}, reward={total_reward}, miss_optimal={optimal_misses[-1]}")

    return model, rewards_log, optimal_misses, founded_ep