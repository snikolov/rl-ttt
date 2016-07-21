import numpy as np

from agent import TDAgent
from environment import State


def test_step_first():
    # Test case when there is no previous state and action. Test that
    # previous state and action are set during step().
    agent = TDAgent(0)
    state = State()
    assert agent.prev_state is None
    assert agent.prev_action is None
    action = agent.step(State(), 0, False)
    assert agent.prev_state == state
    assert agent.prev_action == action


def test_step_normal_Q_learning():
    # Test case when there is a previous state and action but the game
    # is not done. Check that Q is updated correctly in the
    # deterministic greedy case (epsilon=0).

    alpha = 0.1
    gamma = 0.9
    agent = TDAgent(0, alpha=alpha, gamma=gamma, epsilon0=0, method='q-learning')
    agent.prev_action = (1,1)
    agent.prev_state = State()
    prev_afterstate = agent.prev_state.put(agent.prev_action, 0)

    # Construct a Q function to force a spefic update.
    # Value before learning.
    q_prev = 0.6
    agent.Q[prev_afterstate] = q_prev
    curr_state = State().put((1,1), 0).put((0,0), 1)
    # Create two possible actions, one better than the other. The
    # agent should choose to use the value of taking action (2,2) for
    # the target value.
    agent.Q[curr_state.put((2,2), 0)] = 0.7
    agent.Q[curr_state.put((0,2), 0)] = 0.5
    future_return = gamma * 0.7

    # Take a step.  We are not done, but give a nonzero reward to
    # check that it is used.
    reward = 1
    done = False
    action = agent.step(curr_state, reward, done)

    # Value after learning
    q_curr = agent.Q[prev_afterstate]
    print q_prev, q_curr

    assert q_curr == q_prev + alpha * (
        reward + future_return - q_prev
    )


def test_step_normal_Q_sarsa():
    # Test case when there is a previous state and action but the game
    # is not done. Check that Q is updated correctly in the
    # deterministic greedy case (epsilon=0).

    alpha = 0.1
    gamma = 0.9
    agent = TDAgent(0, alpha=alpha, gamma=gamma, epsilon0=0, method='sarsa')
    agent.prev_action = (1,1)
    agent.prev_state = State()
    prev_afterstate = agent.prev_state.put(agent.prev_action, 0)

    # Construct a Q function to force a spefic update.
    # Value before learning.
    q_prev = 0.6
    agent.Q[prev_afterstate] = q_prev
    curr_state = State().put((1,1), 0).put((0,0), 1)
    # Create two possible actions, one better than the other. The
    # agent should choose to use the value of taking action (2,2) for
    # the target value.
    agent.Q[curr_state.put((2,2), 0)] = 0.7
    agent.Q[curr_state.put((0,2), 0)] = 0.5
    future_return = gamma * 0.7

    # Take a step.  We are not done, but give a nonzero reward to
    # check that it is used.
    reward = 1
    done = False
    action = agent.step(curr_state, reward, done)

    # Value after learning
    q_curr = agent.Q[prev_afterstate]

    assert q_curr == q_prev + alpha * (
        reward + future_return - q_prev
    )


def test_step_done():
    # Test case where game is done. Check that action return is None
    # and that end-of-episode adjustments to parameters are made.

    alpha = 0.1
    gamma = 0.9
    alpha_decay = 0.99
    epsilon = 1.0
    epsilon_decay = 0.8
    agent = TDAgent(
        0,
        alpha=alpha,
        gamma=gamma,
        alpha_decay=alpha_decay,
        epsilon0=epsilon,
        epsilon_decay=epsilon_decay)

    agent.prev_state = State()
    agent.prev_action = (1,1)

    state = State()
    reward = 1
    done = True

    action = agent.step(state, reward, done)
    assert action == None
    assert agent.epsilon == epsilon * epsilon_decay
    assert agent.alpha == alpha * alpha_decay
    assert agent.gamma == gamma


def test_act_one_max():
    # Test the case of one maximum value and corresponding action.
    agent = TDAgent(0, epsilon0=0)
    agent.Q[State().put((1, 1), 0)] = 0.6
    agent.Q[State().put((0, 0), 0)] = 0.8
    assert agent._act(State()) == (0, 0)


def test_act_two_max():
    # Test the case of two maximum values and corresponding actions.
    agent = TDAgent(0, epsilon0=0)
    agent.Q[State().put((1, 1), 0)] = 0.6
    agent.Q[State().put((0, 0), 0)] = 0.6
    action = agent._act(State())
    assert action == (1, 1) or action == (0, 0)


def test_act_no_actions():
    # No actions available.
    agent = TDAgent(0, epsilon0=0)
    action = agent._act(State(((1,1,1), (1,1,1), (1,1,1))))
    assert action is None
