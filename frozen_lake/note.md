# Q-learning

## Algorithm

The weight for a step from a state `delta_t` steps into the future is calculated as `gamma^delta_t * gamma` (the *discount factor*) is a number between 0 and 1 (0 <= `gamma` <= 1) and has the effect of valuing rewards received earlier higher than those received later (reflecting the value of a "good start"). `gamma` may also be interpreted as the probability to succeed (or survive) at every step `delta_t`.

The algorithm, therefore, has a function that calculates the quality of a state-action combination:

```
Q : S x A -> R (I have no idea what does this mean)
```

- `Q` - the Q table (2-dimension matrix)
- `S` - the state "scalar"/"matrix"?
- `A` - the action "scalar"/"matrix"?
- `R` - the reward "matrix"?

Before learning begins, `Q` is initialzed to a possibly arbitrary fixed value (chosen by the programmer). Then, at each time `t` the agent selects an action `a[t]`, observes a reward `r[t]`, enters a new state `s[t+1]` (that may depend on both the previous state `s[t]` and the selected action), and `Q` is updated. The core of the algorithm is a simple value iteration update, using the weighted average of the old value and the new information:

```
Q[s[t], a[t]] = (1 - alpha) * Q[s[t], a[t]] + alpha * (r[t] + gamma * max(Q[s[t+1], a]))
```

- `alpha` - learning rate
- `gamma` - discount factor
- `max(Q[s[t+1], a])` -no idea

where `r[t]` is the reward received when moving from the state `s[t]` to the state `s[t+1]`, and `alpha` is the learning rate (0 < `alpha` <= 1).

An episode of the algorithm ends when state `s[t+1]` is a final or *terminal state*. However, Q-learning can also learn in non-episodic tasks. If the discount factor is lower than 1, the action values are finite even if the problem can contain infinite loops.

For all final states `s[f]`, `Q[f, a]` is never updated, but is set to the reward value `r` observed for state `s[f]`. In most cases, `Q[s[f], a] can be taken to equal zero.