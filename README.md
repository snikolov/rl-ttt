# Setup
Requirements:
- nose>=1.3.7
- numpy>=1.11.1
- matplotlib>=1.3.1

To install:
<pre>pip install -r requirements.txt</pre>

To run tests:
<pre>nosetests</pre>

# Summary
- I trained an agent to play tic-tac-toe using both the Q-Learning and SARSA algorithms. 
- I train the agents by playing them against themselves (Q-learning vs. Q-learning, SARSA vs SARSA) and against a random agent.
- I use the following reward structure: 1 for a win, -1 for a loss, 0 for a draw or any other state.
- I keep track of the value of a state and action using a Q function whose keys are the afterstates resulting from a given state and action.
- To choose actions, I use the epsilon-greedy policy derived from Q.
- I decay the learning rate (alpha) and the greediness parameter (epsilon) to 0 over time (i.e the number of games played).
- To evaluate the agent, I
  - plotted the maximum value action possible from an empty board state vs. episode number.
  - played trained agents against a random agent and observed overwhelming win rate.
  - played trained agents against themselves and observed a high draw rate.
  - played against trained agents myself and observed them "trying" to win and stop me from winning.
- Other features include:
  - saving and loading agents (and pre-trained agents in agents directory)
  - human player mode (see Demo).
  - display of gameplay and value map at each step.

# Results
## Q-Learning trained with self-play
While epsilon is high, the agents are in an exploratory phase, and they estimate that they have some non-zero expected return (i.e. that they have a chance of winning, on average). As the players get better, and as they enter the exploitation phase, they almost always draw, and the expected return goes to 0.
![q-vs-q](https://raw.githubusercontent.com/snikolov/rl-ttt/master/plots/q-vs-q.png)

To reproduce:
<pre>python run.py q-learning q-learning 20000 --epsilon 1.0</pre>

## SARSA trained with self-play
The behavior is the same as with Q-Learning vs Q-Learning.
![sarsa-vs-sarsa](https://raw.githubusercontent.com/snikolov/rl-ttt/master/plots/sarsa-vs-sarsa.png)
To reproduce:
<pre>python run.py sarsa sarsa 20000 --epsilon 1.0</pre>

## Q-Learning trained against a random agent
Unlike training with self-play, we can learn to win most of the time when training against a static random agent.
![q-vs-random](https://raw.githubusercontent.com/snikolov/rl-ttt/master/plots/q-vs-random.png)

To reproduce:
<pre>python run.py q-learning random 0 50000 --epsilon 1.0</pre>

## SARSA trained against a random agent
The results are similar to Q-Learning trained against a random agent.
![sarsa-vs-random](https://raw.githubusercontent.com/snikolov/rl-ttt/master/plots/sarsa-vs-random.png)

To reproduce:
<pre>python run.py sarsa random 0 50000 --epsilon 1.0</pre>

# Demo
Try playing against one of the trained agents!
<pre>python run.py q-learning human 1 --player0-file 'agents/q-vs-random/0.pkl' --epsilon 0 --verbose</pre>

Play a trained agent against a random agent 5 times (with gameplay, state, and value map displayed at each step):
<pre>python run.py q-learning random 5 --player0-file 'agents/q-vs-random/0.pkl' --epsilon 0 --verbose</pre>

Play a trained agent against a random agent 1000 times (with final score displayed):
<pre>python run.py q-learning random 1000 --player0-file 'agents/q-vs-random/0.pkl' --epsilon 0</pre>

Play a q-agent against a q-agent 1000 times (with final score displayed):
<pre>python run.py q-learning q-learning 1000 --player0-file 'agents/q-vs-q/0.pkl' --player1-file 'agents/q-vs-q/1.pkl' --epsilon 0</pre>


# Discussion and future work
## Afterstates
I represent state, action pairs as afterstates and keep track of values of the afterstates. An afterstate is a state of the board immediately after a player moves. Tracking the values of afterstates rather than state and action pairs improves learning. It renders equivalent multiple states and actions that result in the same afterstate, and these state/action pairs are in effect updated together. 

## Rewards
I initially had a reward structure that did not penalize losses. This resulted in players not trying to block the opponent's winning move. Adding a negative reward solved this.
