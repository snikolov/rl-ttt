'''Main script for gameplay and display.'''

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from agent import HumanAgent, RandomAgent, TDAgent
from environment import Environment, State


def print_game_state(curr_player, state, reward, done):
    print '---------------------------------------------------'
    print 'Player', curr_player.id
    print 'State:'
    print state
    print 'Values of actions from current state:'
    if hasattr(curr_player, 'Q'):
        print q_to_array(curr_player.Q, curr_player.id, state)
    print 'Reward =', reward
    print 'Done =', done


def run_episode(verbose=False):
    '''Run through a game (episode) until a win or a draw. Return the ID
    of the winner, or None if there is a draw.'''

    # Choose a random player to start.
    player_id = np.random.randint(2)

    # Reset environment for another game.
    env.reset()

    # Initialize state, reward, and done variables, which will be
    # emitted from environment and sent to agents.
    state = env.state
    reward = 0
    done = False

    while True:
        curr_player = players[player_id]
        if verbose:
            print_game_state(curr_player, state, reward, done)

        action = curr_player.step(state, reward, done)

        if verbose:
            print 'Taking action:', action, '\n'
        state, reward, done, winner = env.step(player_id, action)

        # If the game is over, make the agents learn and break the
        # loop. Give the latest reward emitted by the environment to
        # the player that moved last. If the game is a draw, give the
        # same reward to the other player. If the game is won, give
        # the loser the negative of the reward given to the winner.
        if done:
            curr_player.step(state, reward, done)
            other_player = players[0 if player_id == 1 else 1]
            if winner is not None:
                # Give the loser the negative of the reward given the winner.
                other_player.step(state, -reward, done)
            else:
                # In case of a tie, give the other player the same reward.
                other_player.step(state, reward, done)

            # Print final game state and result.
            if verbose:
                print_game_state(curr_player, state, reward, done)
                if reward > 0:
                    print '\n\n*** PLAYER {} WINS! ***\n\n'.format(player_id)
                else:
                    print '\n\n*** IT\'S A DRAW! ***\n\n'
            break

        # Switch players.
        player_id = 0 if player_id == 1 else 1

    return winner


def q_to_array(Q, player_id, state):
    '''Return 2D array values of afterstates from a given state.'''

    actions = state.empty_cells()
    afterstates = [ state.put(action, player_id)
                    for action in actions ]
    arr = np.zeros((3,3))
    for i in xrange(len(actions)):
        arr[actions[i][0], actions[i][1]] = Q[afterstates[i]]
    return arr


def create_player(player_id, player_type, epsilon):
    '''Create player of a particular type.'''

    if player_type == 'q-learning':
        return TDAgent(player_id, epsilon0=epsilon, method='q-learning')
    elif player_type == 'sarsa':
        return TDAgent(player_id, epsilon0=epsilon, method='sarsa')
    elif player_type == 'random':
        return RandomAgent(player_id)
    elif player_type == 'human':
        return HumanAgent(player_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('player0', help='Type of player 0')
    parser.add_argument('player1', help='Type of player 1')
    parser.add_argument('num_episodes', type=int, help='Number of games to play.')
    parser.add_argument('--epsilon', type=float, default=0.2,
        help='Greediness parameter for all epsilon-greedy agents.')
    parser.add_argument('--player0-file', help='File from which to load player 0.')
    parser.add_argument('--player1-file', help='File from which to load player 1.')
    parser.add_argument('--agent-out-dir', default='agents', help='Directory in which to save agents.')
    parser.add_argument('--save', help='Whether to save agents.', action='store_true')
    parser.add_argument('--verbose', help='Toggles detailed display of gameplay', action='store_true')
    args = parser.parse_args()

    # Create environment and players.
    env = Environment()
    players = [
        create_player(0, args.player0, args.epsilon),
        create_player(1, args.player1, args.epsilon)
    ]

    # Optionally load previously saved state.
    if args.player0_file is not None:
        players[0].load_table(args.player0_file)
    if args.player0_file is not None:
        players[0].load_table(args.player0_file)

    # Run episodes and plot results.
    episodes = []
    p0_max_values = []
    p1_max_values = []

    # Empty board state to be used for computing and visualizing value
    # of actions taken from that state.
    start_state = State()
    plt.ion()

    num_player0_wins = 0
    num_player1_wins = 0
    num_draws = 0
    for episode in xrange(args.num_episodes):
        if episode % 500 == 0:
            print '==================================================='
            print 'EPISODE:', episode
            # Display the value of actions from empty board state.
            episodes.append(episode)
            if type(players[0]) is TDAgent:
                print 'Player 0'
                p0_start_values = q_to_array(players[0].Q, 0, start_state)
                p0_max_value = np.max(p0_start_values)
                p0_max_values.append(p0_max_value)
                print 'Value actions from empty board:'
                print p0_start_values
                print 'Value of optimal action from empty board:', p0_max_value
            if type(players[1]) is TDAgent:
                print 'Player 1'
                p1_start_values = q_to_array(players[1].Q, 1, start_state)
                p1_max_value = np.max(p1_start_values)
                p1_max_values.append(p1_max_value)
                print 'Value of player 1 actions from empty board:'
                print p1_start_values
                print 'Value of optimal action from empty board:', p1_max_value

            # Plot values of optimal actions from an empty board.
            if len(p0_max_values) > 1 or len(p1_max_values) > 1:
                legend = []
                if len(p0_max_values) > 1:
                    legend.append('player0')
                    plt.plot(episodes, p0_max_values, color='b')
                if len(p1_max_values) > 1:
                    legend.append('player1')
                    plt.hold(True)
                    plt.plot(episodes, p1_max_values, color='r')
                    plt.hold(False)
                plt.legend(legend, loc=4)
                plt.xlabel('episode')
                plt.ylabel('max value from empty state')
                plt.title('{} vs {}'.format(args.player0, args.player1))

        # Count wins.
        winner = run_episode(verbose=args.verbose)
        if winner == 0:
            num_player0_wins += 1
        elif winner == 1:
            num_player1_wins += 1
        elif winner is None:
            num_draws += 1


    print '\nScore:'
    print 'Player 0:', num_player0_wins
    print 'Player 1:', num_player1_wins
    print 'Draw:', num_draws

    # Save the agents, if save flag is on and saving is applicable.
    if args.save:
        if type(players[0]) is TDAgent:
            out_path0 = os.path.join(
                args.agent_out_dir,
                'agent-' + datetime.strftime(datetime.now(), '%Y%m%d%H%M%S') + '-0.pkl')
            players[0].save_table(out_path0)

        if type(players[1]) is TDAgent:
            out_path1 = os.path.join(
                args.agent_out_dir,
                'agent-' + datetime.strftime(datetime.now(), '%Y%m%d%H%M%S') + '-1.pkl')
            players[1].save_table(out_path1)

    # Show plot.
    raw_input('Done. Press any key.')
