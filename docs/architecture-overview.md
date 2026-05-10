# Architecture overview — how the four variants relate

This is the cross-cutting map. Each variant has its own deeper docs in
`<variant>/docs/`; this file is the navigation layer between them.

## The 2 × 2 matrix

```
                   ┌──────────────────────────────┬──────────────────────────────┐
                   │  PyTorch / lerobot           │  JAX / serl_launcher         │
                   │  (gym_hil + SurRoL_v2)       │  (hil-serl-sim, MuJoCo)      │
┌──────────────────┼──────────────────────────────┼──────────────────────────────┤
│  V1  sparse      │  v1-pytorch/                 │  v1-jax/                     │
│  reward          │  ─ first improved fork       │  ─ upstream baseline,        │
│                  │  ─ keeps 0/1 success only    │     unmodified GeorgeAuburn  │
│                  │  ─ F6 logging                │     reference                │
├──────────────────┼──────────────────────────────┼──────────────────────────────┤
│  V2  hybrid      │  v2-pytorch/                 │  v2-jax/                     │
│  reward (V2.1)   │  ─ V2.1 reward + F7 logging  │  ─ byte-identical V2.1       │
│                  │  ─ actor.py / training_      │     wrapper port             │
│                  │    logger.py extended        │  ─ JAX trainer untouched     │
└──────────────────┴──────────────────────────────┴──────────────────────────────┘
```

## What's held constant across cells (so comparisons are clean)

- **Task**: Franka Panda pick-and-lift, 10 cm lift threshold.
- **Reward design** within a row: V1 is sparse 0/1 in both cells; V2 is
  the *same* hybrid formula in both cells (PyTorch wrapper and JAX
  wrapper produce byte-identical numbers given the same MuJoCo state).
- **Sim physics**: MuJoCo (PyTorch via gym_hil's `PandaPickCubeGymEnv`,
  JAX via `franka_sim`'s `PandaPickCubeGymEnv` — same XMLs, same
  controller).
- **Discount γ = 0.97**, **`temperature_init = 1.0`**, **REDQ-10
  ensemble with 2-sample subsampling** in the SAC variants.
- **F6 logging** in both PyTorch variants (training + episode CSV +
  optional wandb sync).

## What differs across cells (the variables under study)

| Axis | What changes |
|---|---|
| **Reward (V1 → V2)** | sparse 0/1 → hybrid sparse + small dense shaping, with anti-farming guarantees. See [`v2-pytorch/docs/CHANGES.md`](../v2-pytorch/docs/CHANGES.md) §8 / §8b. |
| **Framework (PyTorch → JAX)** | lerobot's modular SAC with REDQ in PyTorch ↔ serl_launcher's RLPD trainer in JAX/Flax. Different impls, different optimisations, different logging stacks. |
| **Logging schema (V1 → V2.1)** | V2.1 PyTorch adds **F7** per-stage reward sums (`r_sub_*` + `grasp_event_fired`). JAX V2.1 surfaces the same dict via `info["reward_dict"]` but the JAX trainer doesn't yet pipe it into its loggers — see [JAX side TODO](#jax-side-followups). |

## Suggested ablations (with which cells to use)

| Question | Run cells | Compare |
|---|---|---|
| Does V2.1 hybrid reward improve sample efficiency over V1 sparse? | `v1-pytorch` vs `v2-pytorch` | `rolling_policy_only_success_20` vs `Interaction step` |
| Does V2.1 transfer between RL frameworks? | `v2-pytorch` vs `v2-jax` | `rolling_success_rate_50` vs wall-clock time |
| Did our improvements actually fix the upstream's 10% gap? | `v1-jax` (baseline) vs `v2-pytorch` | full sweep |
| Reward-hacking attribution: where does V2.1's reward come from? | `v2-pytorch` only | `r_sub_success` vs `r_sub_approach` ratio over training |

The third row is the headline experiment for the paper.

## File-level navigation

### Per-variant entry points

```
v1-pytorch/lerobot/run_learner.sh        # learner (terminal 1)
v1-pytorch/lerobot/run_actor.sh          # actor (terminal 2)
v1-pytorch/lerobot/train_config_gym_hil_touch.json
v1-pytorch/lerobot/train_config_gym_hil_headless.json

v2-pytorch/lerobot/run_learner.sh        # learner — V2.1 reward
v2-pytorch/lerobot/run_actor.sh
v2-pytorch/lerobot/train_config_gym_hil_touch.json
v2-pytorch/lerobot/train_config_gym_hil_headless.json
v2-pytorch/lerobot/src/lerobot/rl/staged_reward_wrapper.py   # ← the wrapper

v1-jax/hil-serl-sim/run_learner.sh       # upstream baseline
v1-jax/hil-serl-sim/run_actor.sh

v2-jax/hil-serl-sim/run_learner.sh       # JAX with V2.1 reward
v2-jax/hil-serl-sim/run_actor.sh
v2-jax/hil-serl-sim/examples/experiments/pick_cube_sim/staged_reward_wrapper.py
```

### Per-variant docs

```
v1-pytorch/docs/CHANGES.md                # V1 changelog (8 numbered fixes)
v1-pytorch/docs/DEPLOY_GUIDE.md           # how to deploy on a fresh GPU pod
v1-pytorch/docs/4090_1h_cheatsheet.md     # 1-hour quick smoke-run recipe
v1-pytorch/docs/TOUCH_INTERVENTION_GUIDE.md  # touch-haptic teleop wiring
v1-pytorch/docs/实验合作内容.md            # operations playbook for collaborator
v1-pytorch/docs/sim_architecture_alignment.md  # how PyTorch sim mirrors JAX sim
v1-pytorch/docs/参数对比_V1_V2_vs_hil-serl-sim.md  # hyperparameter delta table
v1-pytorch/docs/hil-serl-sim_*.md         # JAX-side analysis notes

v2-pytorch/docs/                          # same set, V2.1-aware updates
                                          #   CHANGES.md adds §8b (V2.0→V2.1)
                                          #   实验合作内容.md adds WandB section
```

The two `docs/` folders share most content. V2-PT's is the freshest;
diff the two for the V1-vs-V2 evolution.

### Cross-variant resources (this consolidation level)

```
docs/architecture-overview.md             # this file
results/4090_smoke_run/                   # V1 + V2 numerical data + figures
academic/                                 # LOCAL ONLY — paper / slides / etc.
```

## JAX-side follow-ups

The JAX V2.1 wrapper (`v2-jax/.../staged_reward_wrapper.py`) emits
`info["reward_dict"]` and `info["episode_reward_breakdown"]` with the
same schema as the PyTorch wrapper, but the serl_launcher trainer does
*not* yet read these dicts into wandb. To close this gap, the relevant
hook is in
`v2-jax/hil-serl-sim/serl_launcher/agents/continuous/sac.py`'s
`update_actor_critic_step` → metrics loop. Pull keys from the trajectory
batch's `info` field and append to the metrics dict the same way RLPD
already pushes loss values.

This wasn't done in the consolidation pass because it would require
re-touching the JAX trainer, which we kept untouched on purpose so that
JAX-vs-PyTorch comparison stays clean. Either approach is fine — flag
this as a small-scope follow-up if/when the next experiment needs JAX
wandb panels broken out by stage.
