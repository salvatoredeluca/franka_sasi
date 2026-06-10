"""
analyze_dataset_hf.py
─────────────────────
Bimanual Franka FR3 — LeRobot dataset validator.
Loads directly from HuggingFace Hub using the LeRobot API.

Usage:
    python analyze_dataset_hf.py --repo_id your-org/your-dataset
    python analyze_dataset_hf.py --repo_id your-org/your-dataset --episode 0
    python analyze_dataset_hf.py --repo_id your-org/your-dataset --full

    # if private repo, login first:
    hf auth login

Dependencies:
    pip install lerobot datasets numpy matplotlib opencv-python
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
from lerobot.datasets.lerobot_dataset import LeRobotDataset



# ── dimension labels ──────────────────────────────────────────────────────────
STATE_LABELS = [
    "q1_L","q2_L","q3_L","q4_L","q5_L","q6_L","q7_L","grip_L",
    "q1_R","q2_R","q3_R","q4_R","q5_R","q6_R","q7_R","grip_R",
]
ACTION_LABELS = [
    "dx_L","dy_L","dz_L","dwx_L","dwy_L","dwz_L","grip_L",
    "dx_R","dy_R","dz_R","dwx_R","dwy_R","dwz_R","grip_R",
]
SEP = "─" * 70


# ── load dataset ──────────────────────────────────────────────────────────────

def load_dataset(repo_id, episodes=None):
    """
    Load a LeRobot dataset from HuggingFace Hub.
    episodes: list of episode indices to load, or None for all.
    """
    print(f"\n  Loading dataset: {repo_id}")
    print(f"  (downloads to ~/.cache/huggingface/lerobot on first run)\n")

    dataset = LeRobotDataset(repo_id, episodes=episodes)
    return dataset


def get_episode_arrays(dataset, ep_idx):
    """
    Extract state and action arrays for a single episode.
    Returns state (T,16), action (T,14), timestamps (T,)
    """
    # get the frame indices for this episode
    ep_data = dataset.episode_data_index
    start = ep_data["from"][ep_idx].item()
    end   = ep_data["to"][ep_idx].item()

    states, actions, timestamps = [], [], []

    for i in range(start, end):
        sample = dataset[i]

        # state
        if "observation.state" in sample:
            s = sample["observation.state"]
        else:
            # try alternate key names
            keys = [k for k in sample.keys() if "state" in k.lower()]
            s = sample[keys[0]] if keys else None

        # action
        if "action" in sample:
            a = sample["action"]
        else:
            keys = [k for k in sample.keys() if "action" in k.lower()]
            a = sample[keys[0]] if keys else None

        # timestamp
        ts = sample["timestamp"] if "timestamp" in sample else i - start

        if s is not None:
            states.append(np.array(s, dtype=np.float32))
        if a is not None:
            actions.append(np.array(a, dtype=np.float32))
        timestamps.append(float(ts))

    state  = np.stack(states)     if states  else None
    action = np.stack(actions)    if actions else None
    ts     = np.array(timestamps)

    return state, action, ts


# ── checks ────────────────────────────────────────────────────────────────────


def check_episode(state, action, ts, ep_idx, verbose=True):
    issues = []
    T = len(state)

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Episode {ep_idx}   ({T} timesteps)")
        print(f"{'─'*50}")

    # ── dimensionality ────────────────────────────────────────────────────────
    if verbose:
        print(f"  State  shape : {state.shape}   (expected (T, 16))")
        print(f"  Action shape : {action.shape}  (expected (T, 14))")

    if state.shape[1] != 16:
        issues.append(f"State dim {state.shape[1]} ≠ 16")
    if action.shape[1] != 14:
        issues.append(f"Action dim {action.shape[1]} ≠ 14")

    # ── NaN / Inf ─────────────────────────────────────────────────────────────
    s_nan = np.isnan(state).sum()
    a_nan = np.isnan(action).sum()
    s_inf = np.isinf(state).sum()
    a_inf = np.isinf(action).sum()

    if verbose:
        print(f"\n  NaN  — state: {s_nan}  action: {a_nan}")
        print(f"  Inf  — state: {s_inf}  action: {a_inf}")

    if s_nan > 0: issues.append(f"NaN in state ({s_nan} values)")
    if a_nan > 0: issues.append(f"NaN in action ({a_nan} values)")
    if s_inf > 0: issues.append(f"Inf in state ({s_inf} values)")
    if a_inf > 0: issues.append(f"Inf in action ({a_inf} values)")

    # ── joint ranges ──────────────────────────────────────────────────────────
    for side, sl in [("Left", slice(0,7)), ("Right", slice(8,15))]:
        joints = state[:, sl]
        j_min, j_max = joints.min(), joints.max()
        if verbose:
            print(f"\n  Joint range {side} [rad]: [{j_min:.3f}, {j_max:.3f}]")
        if j_max > 4.0 or j_min < -4.0:
            issues.append(f"Joint {side} out of expected range ({j_min:.3f}, {j_max:.3f} rad)")

    # ── gripper ───────────────────────────────────────────────────────────────
    for side, idx in [("Left", 6), ("Right", 13)]:
        g = action[:, idx]
        if verbose:
            print(f"  Gripper {side}: [{g.min():.3f}, {g.max():.3f}]  "
                  f"{'✓' if g.min()>=-0.01 and g.max()<=1.01 else '✗'}")
        if g.min() < -0.1 or g.max() > 1.1:
            issues.append(f"Gripper {side} out of [0,1]: [{g.min():.3f},{g.max():.3f}]")
        if g.std() < 1e-3:
            issues.append(f"Gripper {side} never moved (std={g.std():.5f})")

    # ── velocity spikes ───────────────────────────────────────────────────────
    for side, sl in [("Left", slice(0,3)), ("Right", slice(7,10))]:
        mag    = np.linalg.norm(action[:, sl], axis=-1)
        spikes = (mag > 0.5).sum()
        if verbose:
            print(f"  Lin vel {side}: max={mag.max():.4f} m/s  mean={mag.mean():.4f} m/s  "
                  f"{'✓' if spikes==0 else f'✗ {spikes} spikes'}")
        if spikes > 0:
            issues.append(f"Velocity spike {side}: {spikes} frames > 0.5 m/s")

    # ── frozen arm ────────────────────────────────────────────────────────────
    for side, sl in [("Left", slice(0,8)), ("Right", slice(8,16))]:
        if state[:, sl].std(axis=0).max() < 1e-4:
            issues.append(f"State {side} arm FROZEN — robot may not have moved")
            if verbose:
                print(f"\n  [WARN] {side} arm state std ≈ 0 — frozen arm?")

    # ── timestamp continuity ──────────────────────────────────────────────────
    if len(ts) > 1:
        dt = np.diff(ts)
        if verbose:
            print(f"\n  dt — mean: {dt.mean()*1000:.2f} ms  "
                  f"max: {dt.max()*1000:.2f} ms  "
                  f"min: {dt.min()*1000:.2f} ms  (expected ≈ 33ms @ 30Hz)")
        if (dt > 0.1).sum() > 0:
            issues.append(f"Timestamp gap > 100ms: {(dt>0.1).sum()} occurrences")

    if verbose:
        if issues:
            print(f"\n  ⚠  Issues:")
            for iss in issues: print(f"     • {iss}")
        else:
            print(f"\n  ✓  No issues found")

    return issues


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_episode(state, action, ep_idx, save_path=None):
    T = len(state)
    t = np.arange(T)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Episode {ep_idx} — Bimanual FR3 dataset check", fontsize=13)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    # Joint angles — Left arm
    for ci, (side, sl) in enumerate([("Left", slice(0,7)), ("Right", slice(8,15))]):
        ax = fig.add_subplot(gs[0, ci])
        for j in range(7):
            ax.plot(t, state[:, sl.start+j], label=f"q{j+1}", linewidth=0.9)
        ax.set_title(f"Joint angles — {side} [rad]")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); ax.set_xlabel("timestep")

    # Joint angle std per timestep (motion activity)
    for ci, (side, sl) in enumerate([("Left", slice(0,7)), ("Right", slice(8,15))]):
        ax = fig.add_subplot(gs[1, ci])
        activity = state[:, sl].std(axis=1)
        ax.plot(t, activity, color="tab:purple", alpha=0.8)
        ax.set_title(f"Joint std (motion activity) — {side}")
        ax.grid(alpha=0.3); ax.set_xlabel("timestep")

    # Linear velocity magnitude (from action: dx,dy,dz per arm)
    for ci, (side, sl) in enumerate([("Left", slice(0,3)), ("Right", slice(7,10))]):
        ax = fig.add_subplot(gs[2, ci])
        mag = np.linalg.norm(action[:, sl], axis=-1)
        ax.plot(t, mag, color="tab:orange", alpha=0.8)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="spike thr.")
        ax.set_title(f"Linear vel magnitude — {side} [m/s]")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("timestep")

    # Gripper
    ax = fig.add_subplot(gs[3, :])
    ax.plot(t, action[:, 6],  label="gripper Left",  color="tab:blue")
    ax.plot(t, action[:, 13], label="gripper Right", color="tab:red", linestyle="--")
    ax.set_title("Gripper commands (expected in [0, 1])")
    ax.set_ylim(-0.1, 1.1); ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("timestep")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    else:
        plt.show()
    plt.close()


def plot_distributions(all_states, all_actions, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle("Dataset-wide distributions", fontsize=13)

    axes[0].boxplot(
        [all_states[:, i] for i in range(16)], labels=STATE_LABELS,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="navy")
    )
    axes[0].set_title("State distribution per dimension")
    axes[0].set_ylabel("value"); axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].boxplot(
        [all_actions[:, i] for i in range(14)], labels=ACTION_LABELS,
        patch_artist=True,
        boxprops=dict(facecolor="lightsalmon", alpha=0.7),
        medianprops=dict(color="darkred")
    )
    axes[1].set_title("Action distribution per dimension")
    axes[1].set_ylabel("value"); axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    else:
        plt.show()
    plt.close()


# ── dataset-level summary ─────────────────────────────────────────────────────

def dataset_summary(dataset, all_states, all_actions):
    n_ep = dataset.num_episodes
    n_fr = dataset.num_frames

    print(f"\n{SEP}")
    print(f"  DATASET SUMMARY")
    print(SEP)
    print(f"  repo_id        : {dataset.repo_id}")
    print(f"  Episodes       : {n_ep}")
    print(f"  Total frames   : {n_fr}")
    print(f"  Avg ep length  : {n_fr / n_ep:.0f} frames")
    print(f"  FPS            : {dataset.fps}")
    print(f"  Features       : {list(dataset.features.keys())}")

    if all_states is not None:
        print(f"\n  Global state  mean ± std (per dim):")
        for i, lbl in enumerate(STATE_LABELS):
            print(f"    [{i:2d}] {lbl:8s}  "
                  f"mean={all_states[:,i].mean():+8.4f}  "
                  f"std={all_states[:,i].std():7.4f}  "
                  f"[{all_states[:,i].min():+8.4f}, {all_states[:,i].max():+8.4f}]")

        print(f"\n  Global action mean ± std (per dim):")
        for i, lbl in enumerate(ACTION_LABELS):
            print(f"    [{i:2d}] {lbl:8s}  "
                  f"mean={all_actions[:,i].mean():+8.4f}  "
                  f"std={all_actions[:,i].std():7.4f}  "
                  f"[{all_actions[:,i].min():+8.4f}, {all_actions[:,i].max():+8.4f}]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LeRobot HuggingFace dataset analyzer — Bimanual Franka FR3"
    )
    parser.add_argument("--repo_id",  required=True,
                        help="HuggingFace repo id, e.g. your-org/your-dataset")
    parser.add_argument("--episode",  type=int, default=None,
                        help="Inspect a single episode index")
    parser.add_argument("--full",     action="store_true",
                        help="Run checks on ALL episodes (default: first 5)")
    parser.add_argument("--plot",     action=argparse.BooleanOptionalAction, default=True,
                        help="Generate plots (use --no-plot to disable)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Save plots to this directory instead of showing them")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print(f"  LeRobot HF Dataset Analyzer — Bimanual Franka FR3")
    print(f"  repo_id : {args.repo_id}")
    print(SEP)

    # load
    dataset = load_dataset(args.repo_id)
    n_ep    = dataset.num_episodes
    print(f"  Found {n_ep} episodes  |  {dataset.num_frames} total frames  |  {dataset.fps} fps")

    # single episode mode
    if args.episode is not None:
        if args.episode >= n_ep:
            print(f"[error] Episode {args.episode} not found (dataset has {n_ep})")
            return
        state, action, ts = get_episode_arrays(dataset, args.episode)
        if state is None or action is None:
            print(f"[error] Could not extract state/action for episode {args.episode}")
            return
        issues = check_episode(state, action, ts, args.episode, verbose=True)
        if args.plot:
            save = f"{args.save_dir}/episode_{args.episode}_check.png" if args.save_dir else None
            plot_episode(state, action, args.episode, save_path=save)
        return

    # multi-episode mode
    n_check    = n_ep if args.full else min(5, n_ep)
    all_issues = {}
    all_states, all_actions = [], []
    episode_cache = {}

    for i in range(n_check):
        state, action, ts = get_episode_arrays(dataset, i)
        if state is None or action is None:
            print(f"  [skip] Episode {i} — could not extract state/action")
            continue
        issues = check_episode(state, action, ts, i, verbose=(n_check <= 5))
        if issues:
            all_issues[i] = issues
        all_states.append(state)
        all_actions.append(action)
        if n_check <= 3:
            episode_cache[i] = (state, action)

    all_states  = np.concatenate(all_states,  axis=0) if all_states  else None
    all_actions = np.concatenate(all_actions, axis=0) if all_actions else None

    # summary
    dataset_summary(dataset, all_states, all_actions)

    # final report
    print(f"\n{SEP}")
    print("  FINAL REPORT")
    print(SEP)
    if all_issues:
        print(f"  ⚠  Issues in {len(all_issues)}/{n_check} episodes:")
        for ep, iss_list in all_issues.items():
            print(f"\n  Episode {ep}:")
            for iss in iss_list:
                print(f"    • {iss}")
    else:
        print(f"  ✓  All {n_check} episodes passed checks")

    # plots
    if args.plot and all_states is not None:
        save = f"{args.save_dir}/distributions.png" if args.save_dir else None
        plot_distributions(all_states, all_actions, save_path=save)
        if n_check <= 3:
            for i, (s, a) in episode_cache.items():
                save = f"{args.save_dir}/episode_{i}_check.png" if args.save_dir else None
                plot_episode(s, a, i, save_path=save)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()