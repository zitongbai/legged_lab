# Batch experiment scripts

Scripts for launching batches of training runs (e.g. hyper-parameter sweeps)
across the GPUs on a single machine. Each run goes into its own tmux session.

## One-time setup

Machine-specific settings (Python env path, proxy) are **not** hard-coded in
the scripts. They live one level up in `scripts/` so both the interactive
launcher (`scripts/launch/`) and these batch scripts share one config. Copy the
template and fill it in for your machine:

```bash
cp scripts/env.local.sh.example scripts/env.local.sh
$EDITOR scripts/env.local.sh
```

`env.local.sh` is git-ignored. Fill in:

- `VENV_ACTIVATE` — absolute path to your Python env's `bin/activate`
- `PROXY_URL` — proxy for reaching remote assets (leave empty to disable)

Any value can also be passed inline as an environment variable, e.g.
`VENV_ACTIVATE=/path/bin/activate scripts/experiments/sweep_amp.sh ...`.

## sweep_amp.sh — single-parameter AMP sweep

Sweeps ONE AMP parameter over a list of values, one value per GPU, one tmux
session per run, fixed seed so only the swept parameter varies.

```bash
# preview the commands without launching anything
DRY_RUN=1 scripts/experiments/sweep_amp.sh disc_update_interval 1 3 5 8

# launch for real (needs a free GPU per value)
scripts/experiments/sweep_amp.sh disc_update_interval 1 3 5 8
scripts/experiments/sweep_amp.sh disc_learning_rate   1e-4 5e-5 1e-5
scripts/experiments/sweep_amp.sh grad_penalty_scale   10 20 40
scripts/experiments/sweep_amp.sh style_reward_scale   5 3 2
scripts/experiments/sweep_amp.sh task_style_lerp      0.3 0.4 0.5
```

Supported params and their Hydra override paths:

| param                | Hydra override path                                          |
| -------------------- | ----------------------------------------------------------- |
| disc_update_interval | agent.algorithm.amp_cfg.disc_update_interval                |
| disc_learning_rate   | agent.algorithm.amp_cfg.disc_learning_rate                  |
| grad_penalty_scale   | agent.algorithm.amp_cfg.grad_penalty_scale                  |
| style_reward_scale   | agent.algorithm.amp_cfg.amp_discriminator.style_reward_scale|
| task_style_lerp      | agent.algorithm.amp_cfg.amp_discriminator.task_style_lerp   |

Other env overrides: `SEED` (default 42), `MAX_ITER` (default 20000),
`TASK` (default `LeggedLab-Isaac-AMP-Flat-G1-v0`), `DRY_RUN=1`.

### Watching runs

```bash
tmux ls                                  # list sessions
tmux attach -t amp_disc_update_interval_5
tensorboard --logdir logs/rsl_rl/g1_amp  # compare curves across runs
```

> Note: each run trains for `MAX_ITER` iterations (default 20000) and can take
> a while. Verify the Hydra override took effect with a short dry run
> first: `MAX_ITER=2 scripts/experiments/sweep_amp.sh disc_update_interval 5`
> then check `logs/rsl_rl/g1_amp/<run>/params/agent.yaml`.
