#!/usr/bin/env python3
"""Generate the two demand-profile figures used in the manuscript."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.common import (  # noqa: E402
    grid_direct_demand_profile,
    grid_routing_demand_profile,
    isolated_demand_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/demand_profiles")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t_s = np.linspace(0.0, 60.0*60.0, 361)
    t_min = t_s/60.0
    single = isolated_demand_profile()
    direct = grid_direct_demand_profile()
    routing = grid_routing_demand_profile()

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    y = np.array([single(t) for t in t_s])
    ax.plot(t_min, y, label="Each approach")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Demand [veh/h]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_demand_single_intersection.pdf", bbox_inches="tight")
    plt.close(fig)

    direct_total = 8.0*np.array([direct(t) for t in t_s])
    routing_total = 16.0*np.array([routing(t) for t in t_s])
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(t_min, direct_total, label="Direct demand")
    ax.plot(t_min, routing_total, label="Routing demand")
    ax.plot(t_min, direct_total+routing_total, label="Total demand")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Demand [veh/h]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_demand_grid4x4.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
