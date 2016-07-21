'''Tic-tac-toe agent.'''

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import cPickle
import numpy as np
import os


class Agent:
    '''Abstract base class for tic-tac-toe agent.'''
    __metaclass__ = ABCMeta

    def _afterstate(self, state, action):
        return state.put(action, self.id)

    @abstractmethod
    def step(self, state, reward, done):
        return


class HumanAgent(Agent):
    '''An agent that asks user input.'''
    def __init__(self, id):
        self.id = id

    def step(self, state, reward, done):
        '''Return user input action.'''
        input = raw_input('Please enter position as "i,j": ')
        return int(input[0]), int(input[2])


class RandomAgent(Agent):
    '''An agent that makes random moves.'''

    def __init__(self, id):
        self.id = id

    def step(self, state, reward, done):
        actions = state.empty_cells()
        if len(actions) > 0:
            return actions[np.random.randint(len(actions))]
        else:
            return None


class TDAgent(Agent):
    '''A tic tac toe agent that learns using a TD(0) algorithm (either
    SARSA or Q-Learning).'''

    def __init__(self,
                 id,
                 alpha=0.1,
                 gamma=0.98,
                 epsilon0=0.2,
                 epsilon_decay=1.0-5.0e-4,
                 alpha_decay=1.0-5.0e-5,
                 method='q-learning'):
        '''
        Initialize a SARSA or Q-Learning tic-tac-toe agent.

        Arguments
        id - ID of agent and mark used on the board
        alpha - Learning rate.
        gamma - Discount rate.
        epsilon0 - The initial epsilon.
        epsilon_decay - Multiplicative epsilon decay each episode.
        alpha_decay - Multiplicative alpha decay each episode.
        method - "q-learning" or "sarsa".
        '''

        self.id = id
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon0
        self.epsilon_decay = epsilon_decay

        assert method == 'q-learning' or method == 'sarsa', 'Unknown method: ' + method
        self.method = method

        # Keep track of previous action and state in order to learn.
        self.prev_action = None
        self.prev_state = None

        # Initialize table of values. We represent state, action pairs
        # as afterstates and keep track of values of the
        # afterstates. An afterstate is a state of the board
        # immediately after a player moves. Tracking the values of
        # afterstates rather than state and action pairs improves
        # learning. It renders equivalent multiple states and actions
        # that result in the same afterstate, and these state/action
        # pairs are in effect updated together.
        self.Q = defaultdict(lambda: 0)


    def save_table(self, path):
        '''Save the tabular values to a file.'''
        directory = os.path.split(path)[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'w') as f:
            # We cannot pickle a defaultdict, so convert to dict
            # first. This loses all the default values. Upon loading,
            # we have to start with a defaultdict and insert the
            # non-defaultdict that we deserialized.
            cPickle.dump(dict(self.Q), f)


    def load_table(self, path):
        '''Load the tabular values from a file.'''
        with open(path, 'r') as f:
            self.Q = defaultdict(lambda: 0)
            self.Q.update(cPickle.load(f))


    def step(self, state, reward, done):
        '''Chooses an action based on the state and learns.

        Arguments:
        reward - Reward received from previous action.
        state - Current state.
        done - Whether state is a terminal state.
        '''

        # Choose an action. Update the previous state/action value if
        # appropriate.
        action = None
        afterstate_future_return = 0.0
        if not done:
            # Choose an action from Q based on the state.
            action = self._act(state)

            # If this is the first move, we cannot update the value of
            # the previous state and action. Just return the action.
            if self.prev_state is None or self.prev_action is None:
                self.prev_state = state
                self.prev_action = action
                return action
            else:
                # If this is not the first move, we can update the
                # value corresponding the the previous state and
                # move. To do this, we compute the future return, or
                # the return from the current state (and the chosen
                # action*). From the point of view of the previous
                # state, the future return is the value of the current
                # state/action pair, discounted by gamma. We then
                # treat the sum of the reward and the future return as
                # a target for the previous state/action value and
                # move this value closer to the target.
                #
                # *Note that this is action is distinct from the
                # action we choose to actually take, and differs
                # between Q-learning and SARSA. In particular,
                # Q-learning is "off-policy" and uses the greedy
                # action to form a value target. SARSA is "on-policy"
                # and uses the chosen action (in this case
                # epsilon-greedy) to form the target value.

                if self.method == 'sarsa':
                    afterstate = self._afterstate(state, action)
                    afterstate_future_return = self.gamma * self.Q[afterstate]
                elif self.method == 'q-learning':
                    actions = state.empty_cells()
                    # Compute possible afterstates. Randomly permute
                    # so that we do not always choose the same action
                    # if there is a tie between maximum values.
                    afterstate_values = np.random.permutation([
                        self.Q[self._afterstate(state, a)] for a in actions
                    ])
                    afterstate_future_return = self.gamma * np.max(afterstate_values)

        # Learn by updating the previous state and action.
        prev_afterstate = self._afterstate(self.prev_state, self.prev_action)
        self.Q[prev_afterstate] += \
            self.alpha * (reward + afterstate_future_return - self.Q[prev_afterstate])

        self.prev_state = state
        self.prev_action = action

        # End of episode behavior. Adjust parameters that may change
        # over multiple episodes, such as learning rate alpha and
        # epsilon-greediness.'''
        if done:
            self.epsilon *= self.epsilon_decay
            self.alpha *= self.alpha_decay

        return action


    def _act(self, state):
        '''Return an epsilon greedy action based on the current state and
        values. If there are no actions available, return None'''

        actions = state.empty_cells()
        if len(actions) == 0:
            return None

        afterstate_values = [ (action, self.Q[self._afterstate(state, action)])
                              for action in actions ]
        if np.random.random() < self.epsilon:
            action, value = afterstate_values[np.random.randint(len(afterstate_values))]
        else:
            _, max_value = max(afterstate_values, key = lambda x: x[1])
            # Choose randomly among all actions with that value.
            max_actions = [ action for action, value in afterstate_values
                            if value == max_value ]
            action = max_actions[np.random.randint(len(max_actions))]
        return action
