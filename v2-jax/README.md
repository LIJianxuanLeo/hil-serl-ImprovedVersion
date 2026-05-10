# hilserl_sim_v2

> Hybrid sparse + dense shaping fork of [GeorgeAuburn/hilserl_sim_v1](https://github.com/GeorgeAuburn/hilserl_sim_v1)
> for the Franka pick-and-lift task. Reward design mirrors `staged_reward_wrapper.py` v2.1
> from our PyTorch / lerobot fork [`hilserl-surrol-improved-v2`](https://github.com/LIJianxuanLeo/hilserl-surrol-improved-v2).
> Aligned with the engineering scheme `hilserl_抓取任务_稀疏奖励替换为稠密奖励_完整实现流程.docx`
> (2026-05-04).

## What changed vs `hilserl_sim_v1`

This fork is a **single-file change** on top of the v1 baseline (plus this README and the new
wrapper module). All of the JAX agent, RLPD trainer, F6 logging, and HIL-DAgger machinery is
unchanged — the only difference is the reward signal feeding into the critic.

| Aspect | v1 (upstream) | v2 (this repo) |
|--------|--------------|----------------|
| Per-step reward | `0` (sparse — `_compute_reward()` computed but discarded in `step()`) | hybrid: `1.00·r_success + 0.05·r_approach + 0.10·r_grasp + 0.05·r_lift − 0.01·r_penalty`, clipped to [-1.0, 2.0] |
| Success reward | `1` (override) | `1.0` (additive — preserves Q continuity) |
| Approach kernel | n/a | **differential** `clip(d_prev−d_now, ±0.02)` (kills hovering-farming) |
| Grasp kernel | n/a | **event-based**: fires once when `(d<4cm) AND (z>z_init+5mm)` holds for ≥2 steps |
| Lift kernel | n/a | **clipped** `clip(z_block − z_pick_ref, 0, 0.05)` (kills lift-farming) |
| Expected episode reward | 0 or 1 | ~0.0–0.6 (failure) / ~1.2–1.8 (success) |
| Anti-farming guarantees | n/a | hovering / light-touch / excessive-lift all bounded by design |

## Why hybrid (not "pure dense")

The engineering scheme audited a previous "pure dense" v2.0 design (per-step
reward in `[0, 10]`, success bonus `+10`) and identified three reward-hacking
failure modes:

- **靠近刷分 (hovering-farming)** — continuous distance proxies like `1/(1+5d)`
  reward staying near the target.
- **轻触刷分 (light-touch farming)** — smooth grasp ramps reward repeatedly
  approaching the contact threshold without committing.
- **抬高刷分 (excessive-lift farming)** — unbounded `lift / lift_target` rewards
  lifting past the success height.

v2.1 (this repo) replaces all three with anti-farming-by-construction signals:
differential approach, event-based grasp, and clipped lift. The success
component is identical to v1 sparse (`r_success = 1.0`) — shaping is only an
additive signal, never a replacement for the task's true objective.

### Files added / modified

```
hil-serl-sim/examples/experiments/pick_cube_sim/
├── staged_reward_wrapper.py    # NEW — dense reward wrapper (mirror of PyTorch V2)
└── config.py                   # MODIFIED — wraps env with StagedRewardWrapper
```

## Why dense reward

The upstream `PandaPickCubeGymEnv.step()` overwrites whatever reward
`_compute_reward()` returns with binary `1` on success and `0` otherwise. With
`discount=0.97`, asymptotic Q from a sparse 0/1 signal sits around `1/(1−γ) = 33`
**only at the success absorbing state**; off-success Q stays near zero. SAC's
entropy term (≈ 6 with `temperature_init=1.0`) then easily dominates the actor
gradient, so the policy keeps exploring rather than exploiting — exactly what
the collaborator's 220K-step / 10%-success run on AutoDL exhibited.

The V2 wrapper produces per-step reward in `[0, 10]`, which (with the same
discount) puts steady-state Q in roughly `[15, 40]`. That's well above the
entropy term, so the actor's loss is dominated by `−Q(s, π(s))` instead of
`α·H(π(·|s))`, and the policy actually optimises for the task.

## Reward decomposition (v2.1)

```
r_approach = clip(d_prev − d_now, −0.02, +0.02)                # diff, ±2 cm/step
r_grasp    = 1.0  if  (d<4cm AND z>z_init+5mm  for ≥2 steps)   # event, fires once
            and not already_fired
            else 0.0
r_lift     = clip(z_block − z_pick_ref, 0, 0.05)               # 5 cm cap above grasp z
r_success  = 1.0  if info["succeed"] else 0                    # terminal sparse
r_penalty  = 0.0   (placeholder — collision sensor not present in sim)

reward = 1.00·r_success + 0.05·r_approach + 0.10·r_grasp + 0.05·r_lift − 0.01·r_penalty
reward = clip(reward, −1.0, 2.0)
```

The per-stage breakdown is surfaced through `info["reward_dict"]` (per step) and
`info["episode_reward_breakdown"]` (at terminal step), so it can be plotted
alongside `actor/loss` and `critic/predicted_qs` in wandb.

## Running

The training entry-point is unchanged — same as `hilserl_sim_v1`:

```bash
cd hil-serl-sim
# Learner (one terminal)
bash run_learner.sh
# Actor (another terminal, after learner says "gRPC server started")
bash run_actor.sh
```

Headless servers (no DISPLAY): `MUJOCO_GL=egl bash run_learner.sh`.

The wrapper auto-applies whenever the `pick_cube_sim` task config is
selected — no flag needed. To revert to v1's sparse reward for a side-by-side
comparison, comment out the `env = StagedRewardWrapper(...)` line in
`hil-serl-sim/examples/experiments/pick_cube_sim/config.py`.

## What to expect on wandb

If V2 is doing what it should, the following should differ from v1:

| Panel | v1 | v2 (this repo, v2.1) |
|-------|----|----|
| `critic/predicted_qs` | ~0–1, slow climb | climbs to ~3–8 (much smaller than v2.0's [15, 40] — by design) |
| `critic/rewards` (per step) | mostly 0, occasional 1 | small +/− shaping ~±0.001 most steps; ±1 spikes on grasp event / success |
| `actor/actor_loss` | dominated by `α·log_π` | shaping pulls actor toward task while α auto-tunes |
| `episode/r` (no intervention) | 0 or 1 (binary) | continuous; converged policy ~1.2–1.8 per success episode |
| `episode/grasp_event_fired` | n/a | rises from ~0% to ≥80% as policy learns to actually grasp |
| `episode/r_sub_*` columns | n/a | breakdown into success / approach / grasp / lift contributions |
| `rolling_policy_only_success_20` | likely <10% on no-intervention runs | target ≥30% (matches V2.1 PyTorch design) |

A higher Q value is **not** the success criterion. A high
`rolling_policy_only_success_20` paired with `r_sub_success ≫ r_sub_approach`
is the success criterion — it means the policy is winning by completing the
task, not by grinding shaping reward.

## Provenance

- Upstream baseline: <https://github.com/GeorgeAuburn/hilserl_sim_v1>
  (initial commit `de9781d`, "Initial import: hil-serl-sim 仿真与训练代码")
- V2 reward design: ported from
  `hilserl-surrol-improved-v2/lerobot/src/lerobot/rl/staged_reward_wrapper.py`
  (PyTorch / lerobot variant). The math is byte-identical so any
  JAX-vs-PyTorch comparison isolates the framework, not the reward.
- The original PyTorch V2 design doc:
  `hilserl-surrol-improved-v2/docs/CHANGES.md` (V1 → V2 reward redesign rationale).

## License

Inherits the upstream `hilserl_sim_v1` license. The new file
`staged_reward_wrapper.py` is contributed under the same terms.
