from environment import Environment, State

################################################################
# State
################################################################
def test_state_put():
    # Test basic put operation.
    state = State()
    state = state.put((0, 0), 0)
    state = state.put((1, 1), 1)
    for i in xrange(len(state.board)):
        for j in xrange(len(state.board[0])):
            if i == j == 0:
                assert state.board[i][j] == 0
            elif i == j == 1:
                assert state.board[i][j] == 1
            else:
                assert state.board[i][j] is None


def test_state_hash():
    # Check that different state instances that have equivalent
    # elements hash to the same value and are equal.
    s0 = State()
    s1 = State()
    x = {}
    x[s0] = 1
    assert x[s1] == 1
    assert s0 == s1


def test_check_win():
    # Test whether we correctly identify wins. TODO: Could generate
    # all combinations here, but this is a sanity check.

    # None
    s = State(
        ((None, None, None),
         (None, None, None),
         (None, None, None))
    )
    assert not s.check_win(0)
    assert not s.check_win(1)

    # Vertical
    s = State(
        ((0, None, None),
         (0, None, None),
         (0, None, None))
    )
    assert s.check_win(0)
    assert not s.check_win(1)

    # Diagonal 1
    s = State(
        ((1, 0, 0),
         (0, 1, 0),
         (0, 0, 1))
    )
    assert s.check_win(1)
    assert not s.check_win(0)

    # Diagonal 2
    s = State(
        ((0, 0, 1),
         (0, 1, 0),
         (1, 0, 0))
    )
    assert s.check_win(1)
    assert not s.check_win(0)

    # Horizontal
    s = State(
        ((1, 0, 0),
         (1, 1, 1),
         (None, 0, 0))
    )
    assert s.check_win(1)
    assert not s.check_win(0)


def test_empty_cells():
    # Check whether the empty cells (available moves) are correctly
    # computed.

    # 0
    s = State(
        ((1, 1, 1),
         (1, 1, 1),
         (1, 1, 1))
    )
    assert len(s.empty_cells()) == 0

    # 1
    s = State(
        ((1, 1, 1),
         (1, 1, 1),
         (None, 1, 1))
    )
    assert s.empty_cells() == [(2,0)]

    # > 1
    s = State(
        ((1, 1, 1),
         (1, None, 1),
         (None, 1, 1))
    )
    assert set(s.empty_cells()) == set([(2,0), (1,1)])


################################################################
# Environment
################################################################
def test_done_unused_cells():
    # Test output in the case where someone won but there are still
    # unused cells.
    env = Environment()
    env.step(0, (0,0))
    env.step(0, (0,1))
    state, reward, done, winner = env.step(0, (0,2))
    assert state == State(
        ((0, 0, 0),
         (None, None, None),
         (None, None, None))
    )
    assert reward == 1
    assert done
    assert winner == 0


def test_draw():
    # Test output in the case of a draw
    env = Environment()
    env.step(0, (0,0))
    env.step(0, (0,1))
    env.step(1, (0,2))

    env.step(1, (1,0))
    env.step(1, (1,1))
    env.step(0, (1,2))

    env.step(0, (2,0))
    env.step(0, (2,1))
    state, reward, done, winner = env.step(1, (2,2))

    assert state == State(
        ((0,0,1),
         (1,1,0),
         (0,0,1))
    )
    assert reward == 0
    assert done
    assert winner is None


def test_reset():
    env = Environment()
    env.step(0, (0,0))
    env.reset()
    assert env.state == State()



