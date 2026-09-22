"""Plot the observed contact-window timeline for public step seed 700102."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = json.loads((HERE / "summary.json").read_text())["runs"]
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, layout="constrained")
colors = {"left": "#1764ad", "right": "#d97921"}

for column, policy in enumerate(("terrain_v3", "original_gpu")):
    row = next(r for r in RUNS if r["policy"] == policy and r["seed"] == 700102)
    with np.load(HERE / row["trace_file"]) as saved:
        trace = saved["trace"]
        columns = list(saved["columns"])
        policy_time = saved["policy_time"]
        progress = saved["wheel_progress"]

    def series(name):
        return trace[:, columns.index(name)]

    time = series("end_s")
    assert len(trace) == row["info"]["physical_steps"]
    assert np.allclose(np.diff(time), 0.0005, atol=1e-9)
    contact = {
        side: float(time[np.flatnonzero(series(f"{side}_target_contacts") > 0)[0]])
        for side in ("left", "right")
    }
    first_side = min(contact, key=contact.get)
    second_side = max(contact, key=contact.get)
    first = contact[first_side]
    assert abs(first - row["first_target_contact_s"]) < 1e-9
    relative_time = time - first
    shown = (-0.15 <= relative_time) & (relative_time <= 0.45)
    policy_relative_time = policy_time - first
    policy_shown = (-0.15 <= policy_relative_time) & (policy_relative_time <= 0.45)
    yaw = np.rad2deg(series("yaw"))
    limited = series("velocity_limited_mask").astype(np.int64)
    lam = series("lambda")
    ax_progress, ax_yaw, ax_lambda = axes[:, column]

    for wheel, side in enumerate(("left", "right")):
        ax_progress.plot(policy_relative_time[policy_shown], progress[policy_shown, wheel],
                         marker="o", markersize=2.4, linewidth=1.5, color=colors[side],
                         label=f"{side.title()} wheel")
    ax_yaw.plot(relative_time[shown], yaw[shown], color="#574a91", linewidth=1.7)
    for boundary in (-5, 5):
        ax_yaw.axhline(boundary, color="#b92929", linestyle="--", linewidth=0.9)
    if row["first_attitude_failure_s"] is not None:
        breach = float(row["first_attitude_failure_s"])
        if first - 0.15 <= breach <= first + 0.45:
            ax_yaw.scatter([breach - first], [np.interp(breach, time, yaw)],
                           marker="x", color="#b92929", s=60, zorder=5,
                           label="First 5° breach")
            ax_yaw.legend(loc="upper left", fontsize=8, frameon=False)
    ax_lambda.plot(relative_time[shown], lam[shown], color="#263238", linewidth=1.0,
                   label="Shared residual λ")
    for side, bit, shade in (("left", 16, "#ee7474"), ("right", 32, "#bc8ed0")):
        ax_lambda.fill_between(relative_time[shown], -0.14, -0.04,
                               where=(limited[shown] & bit) != 0, step="post",
                               color=shade, alpha=0.85, label=f"{side.title()} wheel speed limit")
    ax_lambda.set_ylim(-0.17, 1.06)
    ax_lambda.legend(loc="upper right", fontsize=8, frameon=False)

    for ax in axes[:, column]:
        ax.axvline(0, color="#1a1a1a", linestyle="--", linewidth=1.0)
        ax.axvline(contact[second_side] - first, color="#67727c", linestyle=":", linewidth=1.2)
        ax.set_xlim(-0.15, 0.45)
        ax.grid(alpha=0.18)
    ax_progress.legend(loc="upper left", fontsize=8, frameon=False)
    ax_progress.set_title(
        f"{'Terrain v3' if policy == 'terrain_v3' else 'Original GPU'} · "
        f"{'success' if row['info']['success'] else 'failure'}\n"
        f"{first_side.title()} contact {first:.4f} s; "
        f"{second_side.title()} contact +{1000 * (contact[second_side] - first):.1f} ms",
        fontsize=11,
    )
    ax_lambda.set_xlabel("Time relative to first target contact (s)")

axes[0, 0].set_ylabel("Wheel progress relative\nto terrain center (m)")
axes[1, 0].set_ylabel("Yaw (degrees)")
axes[2, 0].set_ylabel("Shared residual λ\nwheel speed-limit strip")
fig.suptitle("Public 21 mm step, seed 700102 · single instrumented 160-world replay",
             fontsize=14)
fig.supxlabel(
    "Progress: 50 Hz policy samples. Yaw, λ and limit masks: 2 kHz physics samples. "
    "Dashed vertical = first contact; dotted = second-wheel contact.\n"
    "Numeric or instrumentation perturbations may change outcomes; these observed timelines do not establish a unique cause.",
    fontsize=8,
)
fig.savefig(HERE / "seed_700102_contact_timeline.png", dpi=180)
print(HERE / "seed_700102_contact_timeline.png")
