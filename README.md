# CS 285 — Deep RL (UC Berkeley)

Coursework: policies implemented from scratch, agent by agent.

## Policies / Algorithms

| HW | Policy | Notes |
|----|--------|-------|
| hw1 | Behavior cloning + DAgger | `hw1_imitation/` — supervised imitation, expert-labeled aggregation |
| hw2 | Policy gradient (REINFORCE) | `pg_agent.py` — reward-to-go, neural net baseline, GAE |
| hw3 | DQN | `dqn_agent.py` — double Q, target networks; envs: CartPole, LunarLander, MsPacman |
| hw3 | SAC | `sac_agent.py` — single/clipped double-Q critics, entropy autotuning; envs: InvertedPendulum, HalfCheetah, Hopper |

`homework_spring2026/hw1` — reissued imitation-learning assignment, same structure.

## Layout

- `src/agents/` — policy/value logic
- `src/scripts/` (+ `modal_run*.py`) — training entry points, local and Modal
- `experiments/*.yaml` — per-env hyperparameter configs (hw3)
- `discussion_folder/` — section notes
- `*.pdf` — assignment specs, final project outline

## Running

Each `hwN` is a `uv` project.

```bash
cd hw2 && uv run src/scripts/run.py --help
cd hw3 && uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml
cd hw3 && uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml
```
