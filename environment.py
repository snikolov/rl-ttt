'''Tic-tac-toe environment.'''

from copy import copy
import numpy as np

class Environment:
    '''Represents the tic-tac-toe environment. The environment accepts
    actions, changes internal board state, and outputs the effects of the
    actions, including the new state, the reward received, whether the
    game is finished, and whether there is a winner.'''

    def __init__(self):
        '''Initialize an environment.'''
        self.state = State()
        self.win_reward = 1


    def step(self, player_id, action):
        '''Take action on behalf of player_id and return (state, reward, done)
        after action is completed.'''
        # Perform action by player.
        self.state = self.state.put(action, player_id)

        reward = 0
        has_won = self.state.check_win(player_id)
        winner = None
        if has_won:
            winner = player_id
        done = len(self.state.empty_cells()) == 0 or has_won
        if has_won:
            reward = self.win_reward
        return self.state, reward, done, winner


    def reset(self):
        '''Reset the environment state to the game's start state.'''
        self.state = State()


class State:
    '''Represents an immutable state of the tic-tac-toe board.'''

    def __init__(self, board=None):
        '''Initializes a 3x3 tuple representing board state.

        Arguments:
        board - Optional 3x3 nested tuple representing the
            board state.
        '''
        if board is None:
            arr = [ [None] * 3 ] * 3
            # Convert to tuple for immutability and ease of hashing.
            self.board = tuple([tuple(x) for x in arr])
        else:
            if type(board) is not tuple:
                raise ValueError('Input board is not a tuple!')
            if len(board) != 3 or len(board[0]) != 3:
                raise ValueError('Input board is not 3x3.')
            self.board = board


    def __str__(self):
        B = self.board
        return str(
            np.array([
                [ '.' if B[i][j] is None else str(B[i][j])
                  for j in xrange(len(B[0])) ]
                for i in xrange(len(B)) ]))


    def __hash__(self):
        return self.board.__hash__()


    def __eq__(self, other):
        return self.board == other.board


    def put(self, action, mark):
        '''Returns a new board state with mark inserted in cell i,j. Raises
        ValueError if cell is already occupied.'''
        i, j = action
        if self.board[i][j] is not None:
            raise ValueError('Cell {},{} is already occupied!'.format(i, j))
        ret = [ list(row) for row in self.board ]
        ret[i][j] = mark
        ret = tuple([ tuple(row) for row in ret ])
        return State(ret)


    def empty_cells(self):
        '''Return indices of the empty, or None, cells.'''
        B = np.array(self.board)
        indices_r, indices_c = np.indices(B.shape)
        indices_r = indices_r.reshape(-1)
        indices_c = indices_c.reshape(-1)
        return [ (indices_r[i], indices_c[i]) for i in xrange(len(indices_r))
                 if B[indices_r[i], indices_c[i]] is None ]


    def check_win(self, mark):
        '''Return True if the player associated with the given mark has
        won. Otherwise, return False.'''

        B = self.board
        return (
            # Horizontal
            B[0][0] == B[0][1] == B[0][2] == mark or
            B[1][0] == B[1][1] == B[1][2] == mark or
            B[2][0] == B[2][1] == B[2][2] == mark or
            # Vertical
            B[0][0] == B[1][0] == B[2][0] == mark or
            B[0][1] == B[1][1] == B[2][1] == mark or
            B[0][2] == B[1][2] == B[2][2] == mark or
            # Diagonal
            B[0][0] == B[1][1] == B[2][2] == mark or
            B[2][0] == B[1][1] == B[0][2] == mark
        )




