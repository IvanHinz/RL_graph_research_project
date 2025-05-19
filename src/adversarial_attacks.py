import torch

# here observation, state and features are the same
def fgsm_attack(state, model, epsilon=1): # 0.05
    state = state.clone().detach().requires_grad_(True)

    # for DQN MLP is used, not simple GNN, so here only state/observation/features is used
    q_values = model(state.unsqueeze(0))
    # the action that gives best q(s, a)
    action = torch.argmax(q_values, dim=-1)

    model.zero_grad()
    # define the loss
    loss = q_values[0, action]
    loss.backward()

    # define adversarial state
    adv_state = state + epsilon * state.grad.data.sign()
    return adv_state.detach()

# fast gradient sign method on critic in a2c model (the idea is very similar to eacn, but here is used sign function)
def fgsm_critic(state, model, current_node, epsilon=1):
    state = state.clone().detach().requires_grad_(True)

    # simple GNN layer needs adjacency/current_node and state/observation/features
    _, value = model(state.unsqueeze(0), torch.eye(state.shape[0]), current_node)

    model.zero_grad()
    # define the loss
    loss = -value
    loss.backward()

    # define adversarial state
    adv_state = state + epsilon * state.grad.data.sign()
    return adv_state.detach()

# fast gradient sign method on actor in a2c model
def fgsm_actor(state, model, current_node, epsilon=1):
    state = state.clone().detach().requires_grad_(True)

    logits, _ = model(state.unsqueeze(0), torch.eye(state.shape[0]), current_node)
    # convert output of the actor-critic logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # get the most probable action
    action = torch.argmax(probs, dim=-1)

    model.zero_grad()
    # define the loss
    loss = -probs[action]
    loss.backward()

    # define the adversarial state
    adv_state = state + epsilon * state.grad.data.sign()
    return adv_state.detach()

# adversarial attack on the critic network
def eacn_attack(state, model, current_node, epsilon=1):
    state = state.clone().detach().requires_grad_(True)

    _, value = model(state.unsqueeze(0), torch.eye(state.shape[0]), current_node)

    model.zero_grad()
    # define the loss
    loss = -value
    loss.backward()
    grad = state.grad
    # grad_norm = grad.norm(p=2)

    # define the adversarial state
    adv_state = state + epsilon * grad
    return adv_state.detach()

# adversarial attack on the actor network
def eaan_attack(state, model, current_node, epsilon=10**3):
    state = state.clone().detach().requires_grad_(True)

    logits, _ = model(state.unsqueeze(0), torch.eye(state.shape[0]), current_node)
    # convert output of the actor-critic logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # get the most probable action
    a_d = torch.argmax(probs, dim=-1)

    model.zero_grad()
    # define the loss
    loss_1 = 1.0 - probs[a_d]
    loss_1.backward(retain_graph=True)
    grad1 = state.grad.clone().detach()
    state.grad.zero_()
    loss_2 = probs[a_d]
    loss_2.backward()
    grad2 = state.grad.clone().detach()

    H = grad1 - grad2
    # print(H)
    # H_norm = torch.norm(H, p=2)

    # define the adversarial state
    adv_state = (state + epsilon * H).detach()
    return adv_state