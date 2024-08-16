import copy

def test_q_value(agent, state, action, n_games):
    """This function's purpose is to get an experimental value of the 
    expected discounted reward from some state of the env on which we applied
     a given action and thne follow the given policy"""

    env_copy = copy.deepcopy(agent.env)

    total_rewards = 0

    for game in range(n_games):
        agent.env = copy.deepcopy(env.copy)
        done = False
        discounted_rewards = 0
        cumulated_discount_factor = 1
        while not done:
            next_state, reward, done = agent.step_testing(state)

            discounted_rewards += cumulated_discount_factor * reward
            cumulated_discount_factor *= agent.discount_factor

            next_state = agent.state_preprocess(next_state, agent.state_shape, state, agent.device)

        total_rewards += discounted_rewards

    return total_rewards / n_games



            
    