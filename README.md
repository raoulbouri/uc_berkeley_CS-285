# CS 285 — Deep Reinforcement Learning (UC Berkeley)

> Building the core RL algorithm families from scratch — imitation, policy gradients, value-based control, and off-policy actor-critic — each as its own small, runnable project, and watching the agents actually learn to move.

**Stack:** PyTorch · Gymnasium / MuJoCo · Atari · `uv`
**Algorithms:** Behavior Cloning · DAgger · REINFORCE (policy gradients) · DQN · Soft Actor-Critic

<p align="center">
  <img src="assets/halfcheetah-sac-rollout.gif" width="90%" alt="Soft Actor-Critic policy running on HalfCheetah-v4">
</p>

<p align="center">
  <em>A Soft Actor-Critic policy trained from scratch on <code>HalfCheetah-v4</code>, rolled out greedily at evaluation.
  Peak evaluation return ≈ <strong>4,200</strong> over 1,000-step episodes — a stable, committed forward gait learned end-to-end from a reward that only says "move forward."</em>
</p>

## The problem

Reinforcement learning is deceptively simple to state — maximize reward — and famously fickle to get working. The same algorithm can look brilliant or broken depending on a target-network update rate, an entropy coefficient, or whether the advantage estimator is centered. This repository is a tour through that fragility: each assignment rebuilds a canonical algorithm from the paper up rather than calling a library, so the moving parts are visible and debuggable.

## What's inside

Each homework is a self-contained project (`hwN/`) managed with [`uv`](https://astral.sh) — its own `pyproject.toml`, lockfile, and entry-point scripts.

### `hw1` — Imitation Learning
Behavior cloning as pure supervised learning on expert demonstrations, then **DAgger**, which closes the loop: roll out the current policy, have the expert relabel the states it actually visited, and aggregate. The contrast is the lesson — behavior cloning drifts off the expert's distribution, DAgger drags it back.

### `hw2` — Policy Gradients
REINFORCE built up one variance-reduction idea at a time: reward-to-go instead of full-trajectory return, a learned neural-network **baseline**, and **Generalized Advantage Estimation** to trade bias against variance. Directly optimizing the policy, no value bootstrap required.

### `hw3` — Q-Learning & Actor-Critic
The off-policy half of the course, and where the cheetah above comes from.
- **DQN** with a target network and double-Q targets — solves `CartPole` and `LunarLander`, and scales to raw-pixel Atari (`MsPacman`) through the standard frame-stacking wrappers.
- **Soft Actor-Critic** for continuous control — twin (clipped double-Q) critics, a squashed-Gaussian policy, and automatic entropy-temperature tuning. Trained on `InvertedPendulum`, `HalfCheetah`, and `Hopper`.

Hyperparameters for every run live as version-controlled YAML in `hw3/experiments/`, so a result is reproducible from a single config file.

## Running it

Each `hwN` is a `uv` project — dependencies and the virtual environment are handled for you.

```bash
# Soft Actor-Critic on HalfCheetah (the policy in the clip above)
cd hw3
uv run python src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune.yaml

# DQN on CartPole
uv run python src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml
```

Training writes a checkpoint per run. The clip above is a saved SAC checkpoint replayed in a live MuJoCo viewer (`render_mode="human"`) and screen-recorded.

## Results

| Task | Algorithm | Outcome |
|------|-----------|---------|
| `HalfCheetah-v4` | SAC (auto-tuned α) | ≈ 4,200 eval return, stable forward gait |
| `InvertedPendulum-v4` | SAC | Balanced to the episode horizon |
| `CartPole-v1` | DQN | Solved |
| `LunarLander-v2` | DQN | Solved |
| `MsPacman` | DQN (Atari) | Learns from pixels |

## Notes & limitations

These are course implementations, tuned to *learn the environment*, not to chase state-of-the-art numbers. Runs are single-seed; a rigorous comparison would sweep seeds and report confidence intervals. The interesting engineering is in the agents (`hw3/src/agents/`) and the value/policy networks (`hw3/src/networks/`) — the rest is infrastructure for logging, replay, and rollout.

---

*Coursework for [UC Berkeley CS 285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/). Assignment starter code © the course staff; agent implementations are my own.*
