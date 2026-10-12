from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
import os
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .results_exporter import ResultsExporter
from .html_visual import HTMLVisualizer


@dataclass
class StudyOutputs:
    output_dir: Path
    figures_dir: Path
    prefix: str
    csv_files: list[Path]
    figure_files: list[Path]
    html_files: list[Path]
    max_abs_residual: float


def set_publication_style() -> None:
    """Apply a compact IEEE-compatible Matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save_fig(fig, figures_dir: Path, name: str, *, dpi: int = 600) -> list[Path]:
    """Save publication figures as vector PDF only."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = figures_dir / f"{name}.pdf"
    try:
        fig.tight_layout(pad=0.35)
    except Exception:
        pass
    for ax in fig.axes:
        try:
            ax.grid(False)
            ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.8)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
        except Exception:
            pass
    fig.savefig(out_pdf, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [out_pdf]


def node_label(net, node_id: int) -> str:
    return net.node_id_to_name.get(node_id, f"N{node_id}")


def movement_label(net, movement_id: int) -> str:
    mv = net.movements[movement_id]
    return mv.movement_name or f"L{mv.from_link}->L{mv.to_link}"


def _vehicle_class_share_vectors(sim) -> list[tuple[str, np.ndarray]]:
    """Return named class-share vectors used for theoretical FD curves."""
    names = [vc.name for vc in sim.vehicle_classes]
    m = len(names)
    out: list[tuple[str, np.ndarray]] = []

    ref = np.asarray([vc.reference_share for vc in sim.vehicle_classes], dtype=float)
    if ref.sum() <= 0:
        ref = np.ones(m, dtype=float)
    ref = ref / ref.sum()
    out.append(("reference", ref))

    if "HDV" in names and "CAV" in names and m == 2:
        hdv_idx = names.index("HDV")
        cav_idx = names.index("CAV")
        for cav_share in [0.0, 0.25, 0.5, 0.75, 1.0]:
            v = np.zeros(m, dtype=float)
            v[cav_idx] = cav_share
            v[hdv_idx] = 1.0 - cav_share
            out.append((f"CAV share {cav_share:.2f}", v))
    return out


def compute_cell_fd_diagnostics(sim, history) -> pd.DataFrame:
    """Compute local CTM FD diagnostics for every cell at every saved time.

    This is the correct data for FD plots. It is not signalized boundary outflow.
    The local FD envelope is min(sending/dt, receiving/dt) using the density and
    class composition of the same cell at that time.
    """
    rows: list[dict] = []
    dt = float(sim.dt)
    class_names = list(sim.class_names)

    for t_index, time_s in enumerate(history.time_s):
        x = history.commodity_state[t_index]
        x_class = sim.class_state_from_commodity(x)
        total = x.sum(axis=1)
        shares = sim.cell_class_shares(x_class)
        qmax, w = sim.effective_parameters(x)
        S_total, _ = sim.sending(x)
        R = sim.receiving(x)

        for cell_id in range(sim.net.n_cells):
            link_id = int(sim.net.cell_link_id[cell_id])
            length_m = float(sim.net.cell_length_m[cell_id])
            storage = float(sim.net.cell_storage_veh[cell_id])
            density_vpm = float(total[cell_id]) / length_m if length_m > 0 else 0.0
            sending_vph = float(S_total[cell_id]) * 3600.0 / dt
            receiving_vph = float(R[cell_id]) * 3600.0 / dt
            row = {
                "time_s": float(time_s),
                "time_min": float(time_s) / 60.0,
                "cell_id": int(cell_id),
                "link_id": link_id,
                "cell_index_on_link": int(sim.net.cell_index_on_link[cell_id]),
                "vehicles_total": float(total[cell_id]),
                "density_veh_km": density_vpm * 1000.0,
                "density_ratio": float(total[cell_id]) / storage if storage > 0 else 0.0,
                "sending_vph": sending_vph,
                "receiving_vph": receiving_vph,
                "local_fd_vph": min(sending_vph, receiving_vph),
                "qmax_vph": float(qmax[cell_id]) * 3600.0,
                "wave_speed_mps": float(w[cell_id]),
                "storage_veh": storage,
                "free_flow_speed_mps": float(sim.net.cell_vf_mps[cell_id]),
                "jam_density_veh_km": float(sim.net.cell_jam_density_vpm[cell_id]) * 1000.0,
            }
            for ci, cname in enumerate(class_names):
                row[f"vehicles_{cname}"] = float(x_class[cell_id, ci])
                row[f"share_{cname}"] = float(shares[cell_id, ci])
            rows.append(row)
    return pd.DataFrame(rows)


def theoretical_fd_curves(sim, n_points: int = 301) -> pd.DataFrame:
    """Theoretical triangular CTM FD curves per link and selected class shares."""
    rows: list[dict] = []
    tau_by_class = np.asarray([vc.time_gap_s for vc in sim.vehicle_classes], dtype=float)
    length_by_class = np.asarray([vc.vehicle_length_m for vc in sim.vehicle_classes], dtype=float)

    for link_id, cells in sim.net.link_cells.items():
        cells_arr = np.asarray(cells, dtype=int)
        vf = float(np.mean(sim.net.cell_vf_mps[cells_arr]))
        kj_vpm = float(np.mean(sim.net.cell_jam_density_vpm[cells_arr]))
        rho_values = np.linspace(0.0, kj_vpm, n_points)

        for share_label, shares in _vehicle_class_share_vectors(sim):
            tau_bar = float(np.dot(shares, tau_by_class))
            length_bar = float(np.dot(shares, length_by_class))
            qmax = vf / (vf * tau_bar + length_bar)
            w = length_bar / tau_bar
            q_values = np.minimum.reduce([
                vf * rho_values,
                np.full_like(rho_values, qmax),
                w * np.maximum(0.0, kj_vpm - rho_values),
            ])
            rho_crit = qmax / vf if vf > 0 else math.nan
            rho_cong = kj_vpm - qmax / w if w > 0 else math.nan
            for rho, q in zip(rho_values, q_values):
                rows.append({
                    "link_id": int(link_id),
                    "class_share_case": share_label,
                    "density_veh_km": float(rho) * 1000.0,
                    "flow_vph": float(q) * 3600.0,
                    "qmax_vph": qmax * 3600.0,
                    "wave_speed_mps": w,
                    "free_flow_speed_mps": vf,
                    "jam_density_veh_km": kj_vpm * 1000.0,
                    "rho_crit_veh_km": rho_crit * 1000.0,
                    "rho_cong_veh_km": rho_cong * 1000.0,
                })
    return pd.DataFrame(rows)


def demand_profiles(sim, demand, history) -> pd.DataFrame:
    rows: list[dict] = []
    commodities = getattr(demand, "commodities", sim.commodities)
    profile_by_commodity = getattr(demand, "_profile_by_commodity", {})

    for time_s in history.time_s:
        for c in commodities:
            profiles = profile_by_commodity.get(c.commodity_id, [])
            rate_vph = sum(float(p.profile(float(time_s))) for p in profiles)
            rows.append({
                "time_s": float(time_s),
                "time_min": float(time_s) / 60.0,
                "commodity_id": int(c.commodity_id),
                "origin_link": int(c.origin_link),
                "destination_node": int(c.destination_node),
                "destination_name": str(c.destination_name),
                "class_name": str(c.class_name),
                "requested_rate_vph": rate_vph,
                "requested_veh_step": rate_vph * float(sim.dt) / 3600.0,
            })
    return pd.DataFrame(rows)


def network_timeseries(sim, history) -> pd.DataFrame:
    rows = []
    cumulative_entered = 0.0
    cumulative_exited = 0.0
    dt = float(sim.dt)
    for ti, time_s in enumerate(history.time_s):
        entered = sum(float(v) for v in history.entry_flow[ti].values()) if ti < len(history.entry_flow) else 0.0
        exited = sum(float(v) for v in history.exit_flow[ti].values()) if ti < len(history.exit_flow) else 0.0
        cumulative_entered += entered
        cumulative_exited += exited
        physical = sum(float(v) for v in history.link_vehicle_count[ti].values())
        virtual_queue = (
            float(history.virtual_queue_total[ti])
            if ti < len(getattr(history, "virtual_queue_total", []))
            else 0.0
        )
        rows.append({
            "time_s": float(time_s),
            "time_min": float(time_s) / 60.0,
            "physical_vehicles": physical,
            "virtual_queue": virtual_queue,
            "total_system_vehicles": physical + virtual_queue,
            "entered_veh_step": entered,
            "exited_veh_step": exited,
            "entered_vph": entered * 3600.0 / dt,
            "exited_vph": exited * 3600.0 / dt,
            "cumulative_entered": cumulative_entered,
            "cumulative_exited": cumulative_exited,
            "conservation_residual": float(history.residual[ti]) if ti < len(history.residual) else math.nan,
            "is_flush": bool(history.is_flush[ti]) if ti < len(history.is_flush) else False,
        })
    return pd.DataFrame(rows)


def movement_flows_vph(sim, history) -> pd.DataFrame:
    rows = []
    dt = float(sim.dt)
    for ti, time_s in enumerate(history.time_s):
        flows = history.movement_flow[ti]
        for movement_id, mv in sim.raw_network.movements.items():
            flow = float(flows.get(movement_id, 0.0))
            rows.append({
                "time_s": float(time_s),
                "time_min": float(time_s) / 60.0,
                "movement_id": int(movement_id),
                "movement_name": mv.movement_name or f"L{mv.from_link}->L{mv.to_link}",
                "from_link": int(mv.from_link),
                "to_link": int(mv.to_link),
                "flow_veh_step": flow,
                "flow_vph": flow * 3600.0 / dt,
            })
    return pd.DataFrame(rows)


def routing_probabilities_sampled(sim, history, routing, sample_every: int = 10, min_probability: float = 1e-6) -> pd.DataFrame:
    """Sample routing probabilities for diagnostics without creating huge CSVs."""
    rows = []
    sample_every = max(1, int(sample_every))
    for ti, time_s in enumerate(history.time_s):
        if ti % sample_every != 0 and ti != len(history.time_s) - 1:
            continue
        if hasattr(routing, "update_assignment"):
            try:
                routing.update_assignment(
                    network=sim.raw_network,
                    vehicle_classes=sim.vehicle_classes,
                    time_s=float(time_s),
                    commodity_state=history.commodity_state[ti],
                )
            except Exception:
                pass
        for mid, mv in sim.raw_network.movements.items():
            for c in sim.commodities:
                try:
                    p = float(routing.probability(
                        from_link=mv.from_link,
                        to_link=mv.to_link,
                        class_name=c.class_name,
                        commodity_id=c.commodity_id,
                        destination_node=c.destination_node,
                        time_s=float(time_s),
                    ))
                except TypeError:
                    p = float(routing.probability(
                        from_link=mv.from_link,
                        to_link=mv.to_link,
                        class_name=c.class_name,
                        time_s=float(time_s),
                    ))
                if p >= min_probability:
                    rows.append({
                        "time_s": float(time_s),
                        "time_min": float(time_s) / 60.0,
                        "movement_id": int(mid),
                        "movement_name": mv.movement_name or f"L{mv.from_link}->L{mv.to_link}",
                        "from_link": int(mv.from_link),
                        "to_link": int(mv.to_link),
                        "commodity_id": int(c.commodity_id),
                        "class_name": c.class_name,
                        "origin_link": int(c.origin_link),
                        "destination_node": int(c.destination_node),
                        "destination_name": c.destination_name,
                        "probability": p,
                    })
    return pd.DataFrame(rows)


def integrity_checks(sim, history, fd_diag: pd.DataFrame) -> pd.DataFrame:
    total_by_cell = []
    max_storage_violation = 0.0
    min_cell_vehicles = math.inf
    max_density_ratio = 0.0
    for x in history.commodity_state:
        total = np.sum(x, axis=1)
        min_cell_vehicles = min(min_cell_vehicles, float(np.min(total)))
        max_storage_violation = max(max_storage_violation, float(np.max(total - sim.net.cell_storage_veh)))
        ratio = total / np.maximum(sim.net.cell_storage_veh, 1e-12)
        max_density_ratio = max(max_density_ratio, float(np.max(ratio)))
        total_by_cell.append(float(np.sum(total)))

    max_abs_residual = max((abs(float(r)) for r in history.residual), default=0.0)
    min_virtual_queue = float(np.min(sim.virtual_queue)) if len(sim.virtual_queue) else 0.0
    max_local_fd_minus_qmax = float((fd_diag["local_fd_vph"] - fd_diag["qmax_vph"]).max()) if not fd_diag.empty else 0.0
    max_sending_minus_qmax = float((fd_diag["sending_vph"] - fd_diag["qmax_vph"]).max()) if not fd_diag.empty else 0.0
    max_receiving_minus_qmax = float((fd_diag["receiving_vph"] - fd_diag["qmax_vph"]).max()) if not fd_diag.empty else 0.0

    final_physical = float(np.sum(history.commodity_state[-1])) if history.commodity_state else 0.0
    final_virtual_queue = float(np.sum(sim.virtual_queue)) if len(sim.virtual_queue) else 0.0
    max_step_residual_jump = float(np.max(np.abs(np.diff(history.residual)))) if len(history.residual) > 1 else 0.0

    checks = [
        ("max_abs_conservation_residual_veh", max_abs_residual, 1e-6, max_abs_residual <= 1e-6),
        ("max_step_residual_jump_veh", max_step_residual_jump, 1e-6, max_step_residual_jump <= 1e-6),
        ("min_cell_vehicles", min_cell_vehicles, -1e-7, min_cell_vehicles >= -1e-7),
        ("max_storage_violation_veh", max_storage_violation, 1e-7, max_storage_violation <= 1e-7),
        ("max_density_ratio", max_density_ratio, 1.0 + 1e-7, max_density_ratio <= 1.0 + 1e-7),
        ("min_virtual_queue_final", min_virtual_queue, -1e-7, min_virtual_queue >= -1e-7),
        ("final_physical_vehicles_after_flush", final_physical, math.inf, True),
        ("final_virtual_queue_after_flush", final_virtual_queue, math.inf, True),
        ("max_local_fd_minus_qmax_vph", max_local_fd_minus_qmax, 1e-7, max_local_fd_minus_qmax <= 1e-7),
        ("max_sending_minus_qmax_vph", max_sending_minus_qmax, 1e-7, max_sending_minus_qmax <= 1e-7),
        ("max_receiving_minus_qmax_vph", max_receiving_minus_qmax, 1e-7, max_receiving_minus_qmax <= 1e-7),
    ]
    return pd.DataFrame([
        {"check": name, "value": value, "tolerance_or_bound": tol, "passed": bool(passed)}
        for name, value, tol, passed in checks
    ])


def plot_network_layout(net, figures_dir: Path, name: str = "fig_network_layout") -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for link_id, link in net.links.items():
        u = net.nodes[link.from_node]
        v = net.nodes[link.to_node]
        ax.annotate(
            "",
            xy=(v.x, v.y),
            xytext=(u.x, u.y),
            arrowprops=dict(arrowstyle="->", lw=1.0, shrinkA=9, shrinkB=9),
        )
        ax.text(
            0.5 * (u.x + v.x),
            0.5 * (u.y + v.y),
            f"L{link_id}",
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        )
    for node_id, node in net.nodes.items():
        ntype = str(node.node_type.value).lower()
        if ntype.startswith("signal"):
            marker, color, size = "s", "blue", 110
        elif ntype.startswith("origin"):
            marker, color, size = "o", "limegreen", 75
        else:
            marker, color, size = "o", "red", 75
        ax.scatter([node.x], [node.y], s=size, marker=marker, facecolor=color, edgecolor="black", linewidth=0.8, zorder=3)
        ax.text(node.x, node.y + 0.04 * max(1.0, _network_span(net)), node.node_name or f"N{node_id}", ha="center", va="bottom", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xlabel(r"Position $x$ [m]")
    ax.set_ylabel(r"Position $y$ [m]")
    ax.set_title("Network layout")
    return save_fig(fig, figures_dir, name)


def _network_span(net) -> float:
    xs = [node.x for node in net.nodes.values()]
    ys = [node.y for node in net.nodes.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def plot_demand_profiles(demand_df: pd.DataFrame, figures_dir: Path, name: str = "fig_demand_profiles_vph") -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    if demand_df.empty:
        ax.text(0.5, 0.5, "No demand data", ha="center", va="center")
    else:
        grouped = demand_df.groupby(["time_min", "origin_link", "class_name"], as_index=False)["requested_rate_vph"].sum()
        for (origin, cls), s in grouped.groupby(["origin_link", "class_name"]):
            ax.plot(s["time_min"], s["requested_rate_vph"], label=f"L{origin} {cls}")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Requested demand [veh/h]")
    ax.set_title("External demand profiles")
    ax.grid(False)
    if len(ax.lines) <= 16:
        ax.legend(loc="best", frameon=True, ncol=2)
    return save_fig(fig, figures_dir, name)


def plot_entry_exit(network_df: pd.DataFrame, figures_dir: Path, prefix: str = "fig") -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(network_df["time_min"], network_df["entered_vph"], label="Entered")
    ax.plot(network_df["time_min"], network_df["exited_vph"], label="Exited")
    if network_df["is_flush"].astype(bool).any():
        fs = network_df.loc[network_df["is_flush"].astype(bool), "time_min"].min()
        ax.axvline(fs, linestyle="--", linewidth=1.0)
        ax.text(fs, 0.96, "flush", rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Flow [veh/h]")
    ax.set_title("Network entry and exit flow rates")
    ax.grid(False)
    ax.legend(loc="best", frameon=True)
    paths += save_fig(fig, figures_dir, f"{prefix}_entry_exit_flows_vph")

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(network_df["time_min"], network_df["cumulative_entered"], label="Entered")
    ax.plot(network_df["time_min"], network_df["cumulative_exited"], label="Exited")
    if network_df["is_flush"].astype(bool).any():
        fs = network_df.loc[network_df["is_flush"].astype(bool), "time_min"].min()
        ax.axvline(fs, linestyle="--", linewidth=1.0)
        ax.text(fs, 0.05, "flush", rotation=90, va="bottom", ha="right", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Cumulative vehicles [veh]")
    ax.set_title("Cumulative network entry and exit")
    ax.grid(False)
    ax.legend(loc="best", frameon=True)
    paths += save_fig(fig, figures_dir, f"{prefix}_cumulative_entered_exited")
    return paths


def plot_movement_flows(movement_df: pd.DataFrame, figures_dir: Path, name: str = "fig_movement_flows_vph", max_movements: int = 12) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    if movement_df.empty:
        ax.text(0.5, 0.5, "No movement flow data", ha="center", va="center")
    else:
        totals = movement_df.groupby("movement_id")["flow_vph"].max().sort_values(ascending=False)
        selected_ids = list(totals.head(max_movements).index)
        for mid in selected_ids:
            s = movement_df[movement_df["movement_id"] == mid]
            label = str(s["movement_name"].iloc[0]).replace(": ", " ")
            ax.plot(s["time_min"], s["flow_vph"], label=label)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Movement flow [veh/h]")
    ax.set_title("Signal-controlled movement flows")
    ax.grid(False)
    if len(ax.lines) <= max_movements:
        ax.legend(loc="best", frameon=True, ncol=2, fontsize=6)
    return save_fig(fig, figures_dir, name)


def plot_network_accumulation(network_df: pd.DataFrame, figures_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(network_df["time_min"], network_df["physical_vehicles"])
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Vehicles [veh]")
    ax.set_title("Network accumulation")
    ax.grid(False)
    paths += save_fig(fig, figures_dir, "fig_network_accumulation")

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(network_df["time_min"], network_df["conservation_residual"])
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Residual [veh]")
    ax.set_title("Conservation residual")
    ax.grid(False)
    paths += save_fig(fig, figures_dir, "fig_conservation_residual")
    return paths


def _plot_theoretical_curves_for_link(ax, theo_df: pd.DataFrame, link_id: int, cases: Sequence[str] = ("reference",)) -> None:
    for case in cases:
        s = theo_df[(theo_df["link_id"] == link_id) & (theo_df["class_share_case"] == case)]
        if not s.empty:
            ax.plot(s["density_veh_km"], s["flow_vph"], linewidth=1.5, label=case)



def _sample_plot_points(df: pd.DataFrame, max_points: int = 350) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, int(math.ceil(len(df) / max_points)))
    return df.iloc[::step].copy()


def plot_theoretical_fd(theo_df: pd.DataFrame, figures_dir: Path, name: str = "fig_theoretical_triangular_fd") -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    if not theo_df.empty:
        link_id = int(theo_df["link_id"].iloc[0])
        for case in ["CAV share 0.00", "reference", "CAV share 1.00"]:
            s = theo_df[(theo_df["link_id"] == link_id) & (theo_df["class_share_case"] == case)]
            if not s.empty:
                ax.plot(s["density_veh_km"], s["flow_vph"], label=case)
    ax.set_xlabel("Density [veh/km]")
    ax.set_ylabel("Flow [veh/h]")
    ax.set_title("Theoretical triangular FD")
    ax.grid(False)
    ax.legend(loc="best", frameon=True)
    return save_fig(fig, figures_dir, name)


def plot_fd_pages(fd_diag: pd.DataFrame, theo_df: pd.DataFrame, figures_dir: Path, *, prefix: str, y_col: str, ylabel: str, title: str, max_links_per_page: int = 16, overlay_theoretical: bool = True, scatter_alpha: float = 0.25, plot_points: bool = True) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    link_ids = sorted(fd_diag["link_id"].unique()) if not fd_diag.empty else sorted(theo_df["link_id"].unique())
    if not link_ids:
        return paths
    for page_start in range(0, len(link_ids), max_links_per_page):
        subset = link_ids[page_start:page_start + max_links_per_page]
        cols = min(4, len(subset))
        rows = int(math.ceil(len(subset) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.45 * rows), squeeze=False)
        for ax, link_id in zip(axes.ravel(), subset):
            s = fd_diag[fd_diag["link_id"] == link_id]
            s_plot = _sample_plot_points(s, max_points=350)
            if plot_points and (not s_plot.empty) and y_col in s_plot:
                ax.scatter(s_plot["density_veh_km"], s_plot[y_col], s=4, alpha=scatter_alpha, linewidths=0, rasterized=True)
            if overlay_theoretical:
                _plot_theoretical_curves_for_link(ax, theo_df, int(link_id), cases=("reference",))
            ax.set_title(f"Link {int(link_id)}")
            ax.set_xlabel("Density [veh/km]")
            ax.set_ylabel(ylabel)
            ax.grid(False)
        for ax in axes.ravel()[len(subset):]:
            ax.axis("off")
        page_no = page_start // max_links_per_page + 1
        paths += save_fig(fig, figures_dir, f"{prefix}_page_{page_no}")
    return paths


def plot_sending_receiving_pages(fd_diag: pd.DataFrame, theo_df: pd.DataFrame, figures_dir: Path, max_links_per_page: int = 16) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    link_ids = sorted(fd_diag["link_id"].unique())
    for page_start in range(0, len(link_ids), max_links_per_page):
        subset = link_ids[page_start:page_start + max_links_per_page]
        cols = min(4, len(subset))
        rows = int(math.ceil(len(subset) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.45 * rows), squeeze=False)
        for ax, link_id in zip(axes.ravel(), subset):
            s = fd_diag[fd_diag["link_id"] == link_id]
            s_plot = _sample_plot_points(s, max_points=350)
            ax.scatter(s_plot["density_veh_km"], s_plot["sending_vph"], s=4, alpha=0.20, linewidths=0, label="sending", rasterized=True)
            ax.scatter(s_plot["density_veh_km"], s_plot["receiving_vph"], s=4, alpha=0.20, linewidths=0, label="receiving", rasterized=True)
            _plot_theoretical_curves_for_link(ax, theo_df, int(link_id), cases=("reference",))
            ax.set_title(f"Link {int(link_id)}")
            ax.set_xlabel("Density [veh/km]")
            ax.set_ylabel("Capacity [veh/h]")
            ax.grid(False)
        for ax in axes.ravel()[len(subset):]:
            ax.axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3, frameon=True)
        page_no = page_start // max_links_per_page + 1
        paths += save_fig(fig, figures_dir, f"fig_sending_receiving_per_link_page_{page_no}")
    return paths


def realized_interior_flow_df(history) -> pd.DataFrame:
    rows = []
    for ti, time_s in enumerate(history.time_s):
        records_by_time = getattr(history, "interior_flow_records", [])
        records = records_by_time[ti] if ti < len(records_by_time) else []
        for r in records:
            rows.append({"time_s": float(time_s), "time_min": float(time_s) / 60.0, **r})
    return pd.DataFrame(rows)


def plot_realized_flow_diagnostic(realized_df: pd.DataFrame, theo_df: pd.DataFrame, figures_dir: Path, max_links_per_page: int = 16) -> list[Path]:
    """Plot realized interior boundary flow. This is useful but not a pure FD."""
    if realized_df.empty:
        return []
    tmp = realized_df.rename(columns={"upstream_density_veh_km": "density_veh_km"})
    return plot_fd_pages(
        tmp,
        theo_df,
        figures_dir,
        prefix="fig_realized_interior_flow_diagnostic_not_fd",
        y_col="flow_vph",
        ylabel="Realized flow [veh/h]",
        title="Realized interior boundary flow diagnostic",
        max_links_per_page=max_links_per_page,
        overlay_theoretical=True,
        scatter_alpha=0.25,
    )


def make_figures_report(output_dir: Path, figures_dir: Path, prefix: str, figure_files: list[Path], checks_df: pd.DataFrame) -> Path:
    report = output_dir / f"{prefix}_figures_report.html"
    pdfs = [p for p in figure_files if p.suffix.lower() == ".pdf"]
    max_res = checks_df.loc[checks_df["check"] == "max_abs_conservation_residual_veh", "value"]
    max_res_str = f"{float(max_res.iloc[0]):.6e}" if not max_res.empty else "n/a"
    failed = checks_df[~checks_df["passed"].astype(bool)] if not checks_df.empty else pd.DataFrame()
    html_doc = """<!doctype html><html><head><meta charset='utf-8'><title>MIXSim Study Report</title>
<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:30px}img{max-width:1180px;width:100%;border:1px solid #cbd5e1;border-radius:8px;background:white;margin:12px 0}.card{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin-bottom:16px}.warn{background:#fff7ed;border-color:#fed7aa}.bad{background:#fef2f2;border-color:#fecaca}code{background:#e2e8f0;padding:2px 4px;border-radius:4px}</style>
</head><body>"""
    html_doc += f"<h1>{html.escape(prefix)} study report</h1>"
    html_doc += f"<div class='card'><b>Max absolute conservation residual:</b> {max_res_str} veh</div>"
    html_doc += "<div class='card warn'><b>FD definition:</b> triangular FD figures use local CTM sending/receiving envelopes and theoretical curves. Realized interior boundary flow is plotted separately as a diagnostic because it can fall below the FD due to downstream receiving constraints.</div>"
    if not failed.empty:
        html_doc += "<div class='card bad'><b>Failed integrity checks:</b><ul>"
        for _, r in failed.iterrows():
            html_doc += f"<li>{html.escape(str(r['check']))}: {float(r['value']):.6g}</li>"
        html_doc += "</ul></div>"
    for p in pdfs:
        rel = os.path.relpath(p, report.parent)
        html_doc += f"<div class='card'><h2>{html.escape(p.stem)}</h2><a href='{html.escape(rel)}'>{html.escape(rel)}</a></div>"
    html_doc += "</body></html>"
    report.write_text(html_doc, encoding="utf-8")
    return report



def _sum_dicts(dict_list):
    """Sum a list of dictionaries with numeric values."""
    out = {}
    for d in dict_list:
        for k, v in (d or {}).items():
            out[k] = float(out.get(k, 0.0)) + float(v or 0.0)
    return out


def _sum_nested_flow_dicts(dict_list):
    """Sum dictionaries whose values are lists of numeric commodity flows."""
    out = {}
    for d in dict_list:
        for k, vals in (d or {}).items():
            vals = list(vals or [])
            if k not in out:
                out[k] = [0.0] * len(vals)
            if len(out[k]) < len(vals):
                out[k].extend([0.0] * (len(vals) - len(out[k])))
            for i, v in enumerate(vals):
                out[k][i] += float(v or 0.0)
    return out


def _sum_interior_records(record_lists):
    """Aggregate interior-flow diagnostic records over a downsample interval."""
    acc = {}
    for records in record_lists:
        for r in records or []:
            key = (r.get("link_id"), r.get("boundary_index"), r.get("upstream_cell_id"), r.get("downstream_cell_id"))
            if key not in acc:
                acc[key] = dict(r)
                acc[key]["flow_veh_step"] = 0.0
                acc[key]["flow_vph"] = 0.0
            acc[key]["flow_veh_step"] += float(r.get("flow_veh_step", 0.0) or 0.0)
            # flow_vph is recomputed by the visual/report tools when needed; keep a summed placeholder.
            acc[key]["flow_vph"] += float(r.get("flow_vph", 0.0) or 0.0)
    return list(acc.values())


def downsample_history(history, stride: int = 1):
    """Return a downsampled SimulationHistory for lighter HTML files.

    State variables are sampled at the retained time indices. Flow variables are
    aggregated over the skipped CTM steps between two retained indices. This is
    important for standalone HTML dashboards: cumulative entered/exited totals
    remain equal to the full-resolution simulation totals even when only a small
    number of animation frames are retained.
    """
    stride = max(1, int(stride))
    if stride == 1:
        # Attach interval durations for visualizer consistency.
        if not hasattr(history, "interval_duration_s"):
            dt = 1.0
            if len(history.time_s) > 1:
                dt = float(history.time_s[1] - history.time_s[0])
            history.interval_duration_s = [0.0] + [dt] * (len(history.time_s) - 1)
        return history

    cls = type(history)
    out = cls()
    n = len(history.time_s)
    if n == 0:
        return out

    keep = list(range(0, n, stride))
    if keep[-1] != n - 1:
        keep.append(n - 1)

    sampled_fields = [
        "time_s", "cell_state", "commodity_state", "link_vehicle_count",
        "link_density_ratio", "residual", "is_flush", "virtual_queue_total",
        "signal_green", "active_phase_by_node", "routing_probabilities",
    ]
    for field in sampled_fields:
        if hasattr(history, field):
            values = getattr(history, field)
            try:
                setattr(out, field, [values[i] for i in keep if i < len(values)])
            except Exception:
                setattr(out, field, values)

    flow_fields = ["movement_flow", "entry_flow", "exit_flow", "interior_flow"]
    nested_flow_fields = ["movement_flow_by_commodity", "entry_flow_by_commodity", "exit_flow_by_commodity"]

    interval_duration_s = []
    prev = 0
    for j, idx in enumerate(keep):
        if j == 0:
            # State at t=0 has no preceding interval.
            interval_indices = [0]
            duration = 0.0
        else:
            # Flow saved at index r is the flow during (time[r-1], time[r]].
            interval_indices = list(range(prev + 1, idx + 1))
            duration = float(history.time_s[idx] - history.time_s[prev])
        interval_duration_s.append(duration)

        for field in flow_fields:
            if hasattr(history, field):
                values = getattr(history, field)
                current = getattr(out, field)
                current.append(_sum_dicts([values[i] for i in interval_indices if i < len(values)]))

        for field in nested_flow_fields:
            if hasattr(history, field):
                values = getattr(history, field)
                current = getattr(out, field)
                current.append(_sum_nested_flow_dicts([values[i] for i in interval_indices if i < len(values)]))

        if hasattr(history, "interior_flow_records"):
            values = getattr(history, "interior_flow_records")
            out.interior_flow_records.append(_sum_interior_records([values[i] for i in interval_indices if i < len(values)]))

        prev = idx

    out.interval_duration_s = interval_duration_s
    return out


def truncate_history(history, stop_index: int):
    """Return a copy of a SimulationHistory through ``stop_index`` (inclusive).

    The helper is intended for visualization/export workflows that simulate a
    generous clearance horizon but should not keep long empty tails after the
    network has cleared.
    """
    stop_index = int(stop_index)
    if stop_index < 0:
        raise ValueError("stop_index must be non-negative")
    cls = type(history)
    out = cls()
    for field_name in getattr(history, "__dataclass_fields__", {}):
        values = getattr(history, field_name)
        if isinstance(values, list):
            setattr(out, field_name, values[: stop_index + 1])
        else:
            setattr(out, field_name, values)
    if hasattr(history, "interval_duration_s"):
        out.interval_duration_s = list(history.interval_duration_s[: stop_index + 1])
    return out


def history_through_clearance(sim, history, *, demand_end_s: float, clearance_tol: float = 1.0e-4):
    """Trim history at the first post-demand state below the system-clearance tolerance."""
    ts = network_timeseries(sim, history)
    total = ts["physical_vehicles"].astype(float) + ts["virtual_queue"].astype(float)
    candidates = ts.index[
        (ts["time_s"].astype(float) >= float(demand_end_s))
        & (total <= float(clearance_tol))
    ].tolist()
    if not candidates:
        return history
    return truncate_history(history, int(candidates[0]))

def run_publication_exports(
    *,
    sim,
    history,
    demand,
    routing,
    signal_controller,
    output_dir: Path | str,
    prefix: str,
    vehicle_class_names: list[str] | None = None,
    routing_sample_every: int = 10,
    max_links_per_page: int = 16,
    html_stride: int = 1,
    include_sending_receiving: bool = True,
    include_realized_diagnostic: bool = True,
    skip_html: bool = False,
    local_fd_points: bool = True,
) -> StudyOutputs:
    """Generate the full reproducible output suite for a network study.

    This creates CSVs, publication-ready PDF/PNG figures, integrity checks, the
    original HTML visualizer, and one HTML figure report. Results are generated
    only when the study script is run; this helper itself is code-only.
    """
    set_publication_style()
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_files: list[Path] = []
    figure_files: list[Path] = []
    html_files: list[Path] = []

    print("[study_tools] exporting core CSV files...")
    # Core simulator exports.
    ResultsExporter(sim, history).export_all(prefix=str(output_dir / prefix))
    csv_files.extend(sorted(output_dir.glob(f"{prefix}_*.csv")))

    print("[study_tools] computing diagnostic tables...")
    # Additional diagnostics.
    fd_diag = compute_cell_fd_diagnostics(sim, history)
    theo = theoretical_fd_curves(sim)
    demand_df = demand_profiles(sim, demand, history)
    network_df = network_timeseries(sim, history)
    movement_df = movement_flows_vph(sim, history)
    realized_df = realized_interior_flow_df(history)
    routing_df = routing_probabilities_sampled(sim, history, routing, sample_every=routing_sample_every)
    checks_df = integrity_checks(sim, history, fd_diag)

    extra_tables = {
        f"{prefix}_cell_fd_diagnostics.csv": fd_diag,
        f"{prefix}_theoretical_fd_curves.csv": theo,
        f"{prefix}_demand_profiles.csv": demand_df,
        f"{prefix}_network_timeseries.csv": network_df,
        f"{prefix}_movement_flows_vph.csv": movement_df,
        f"{prefix}_realized_interior_flow_diagnostic.csv": realized_df,
        f"{prefix}_routing_probabilities_sampled.csv": routing_df,
        f"{prefix}_integrity_checks.csv": checks_df,
    }
    for filename, df in extra_tables.items():
        path = output_dir / filename
        df.to_csv(path, index=False)
        csv_files.append(path)

    print("[study_tools] generating figures...")
    # Figures.
    print("[study_tools]   layout")
    figure_files += plot_network_layout(sim.raw_network, figures_dir)
    print("[study_tools]   demand")
    figure_files += plot_demand_profiles(demand_df, figures_dir)
    print("[study_tools]   entry/exit")
    figure_files += plot_entry_exit(network_df, figures_dir)
    print("[study_tools]   movement flows")
    figure_files += plot_movement_flows(movement_df, figures_dir)
    print("[study_tools]   accumulation/residual")
    figure_files += plot_network_accumulation(network_df, figures_dir)
    print("[study_tools]   theoretical FD")
    figure_files += plot_theoretical_fd(theo, figures_dir)
    print("[study_tools]   local FD per link")
    figure_files += plot_fd_pages(
        fd_diag,
        theo,
        figures_dir,
        prefix="fig_local_ctm_fd_per_link",
        y_col="local_fd_vph",
        ylabel="Local CTM FD flow [veh/h]",
        title="Local CTM FD per link",
        max_links_per_page=max_links_per_page,
        overlay_theoretical=True,
        scatter_alpha=0.22,
        plot_points=bool(local_fd_points),
    )
    if include_sending_receiving:
        print("[study_tools]   sending/receiving")
        figure_files += plot_sending_receiving_pages(fd_diag, theo, figures_dir, max_links_per_page=max_links_per_page)
    else:
        print("[study_tools]   sending/receiving plots skipped (CSV still exported)")
    if include_realized_diagnostic:
        print("[study_tools]   realized flow diagnostic")
        figure_files += plot_realized_flow_diagnostic(realized_df, theo, figures_dir, max_links_per_page=max_links_per_page)
    else:
        print("[study_tools]   realized flow diagnostic plots skipped (CSV still exported)")

    if skip_html:
        print("[study_tools] HTML visualizer skipped by user option")
    else:
        print("[study_tools] building HTML visualizer...")
        # Original rich HTML visualizer, with optional downsampling for long 4x4 runs.
        visualizer_path = output_dir / f"{prefix}_visualization.html"
        HTMLVisualizer(
            sim.raw_network,
            downsample_history(history, html_stride),
            routing_controller=routing,
            signal_controller=signal_controller,
            vehicle_class_names=vehicle_class_names or list(sim.class_names),
            dt=float(sim.dt),
        ).save(visualizer_path)
        html_files.append(visualizer_path)

    print("[study_tools] writing figures report...")
    report = make_figures_report(output_dir, figures_dir, prefix, figure_files, checks_df)
    html_files.append(report)

    max_abs_residual = float(checks_df.loc[checks_df["check"] == "max_abs_conservation_residual_veh", "value"].iloc[0])
    print(f"Saved study outputs to {output_dir}")
    print(f"Max |conservation residual| = {max_abs_residual:.6e} veh")
    return StudyOutputs(
        output_dir=output_dir,
        figures_dir=figures_dir,
        prefix=prefix,
        csv_files=csv_files,
        figure_files=figure_files,
        html_files=html_files,
        max_abs_residual=max_abs_residual,
    )

class IntervalRoutingController:
    """Cache a routing controller and update it only every update_interval_s.

    This is useful for larger grids. The CTM still runs at dt, but shortest-path
    or stochastic routing tables are refreshed at a coarser control interval.
    """

    def __init__(self, base_controller, update_interval_s: float = 30.0):
        self.base_controller = base_controller
        self.update_interval_s = float(update_interval_s)
        self._last_bucket: int | None = None
        if hasattr(base_controller, "class_names"):
            self.class_names = base_controller.class_names

    def _bucket(self, time_s: float) -> int:
        return int(math.floor(float(time_s) / self.update_interval_s + 1e-9))

    def update_assignment(self, *args, **kwargs) -> None:
        time_s = float(kwargs.get("time_s", 0.0))
        if len(args) >= 3:
            # Usually args are network, vehicle_classes, time_s if positional.
            try:
                time_s = float(args[2])
            except Exception:
                pass
        bucket = self._bucket(time_s)
        if bucket != self._last_bucket:
            self._last_bucket = bucket
            if hasattr(self.base_controller, "update_assignment"):
                self.base_controller.update_assignment(*args, **kwargs)

    def probability(self, *args, **kwargs) -> float:
        return float(self.base_controller.probability(*args, **kwargs))

    def commodity_metadata(self) -> list[dict]:
        if hasattr(self.base_controller, "commodity_metadata"):
            return self.base_controller.commodity_metadata()
        return []
