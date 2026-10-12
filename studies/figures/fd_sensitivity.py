#!/usr/bin/env python3
"""
Create IEEE-style composition-dependent fundamental-diagram figures
for MIXSim CAV penetration and CAV time-gap studies.

The script generates two SEPARATE figures:

1. Effect of CAV penetration on the composition-dependent FD.
   CAV penetration rates:
       0%, 25%, 50%, 75%, 100%

2. Effect of the CAV time gap on the composition-dependent FD.
   CAV time gaps:
       0.6, 1.2, 1.8, 2.4, 3.0 s

The penetration study uses the nominal CAV time gap:
    tau_CAV = 0.9 s

The time-gap study keeps the CAV penetration fixed at:
    45%

The HDV time gap is:
    tau_HDV = 3.0 s

The MIXSim triangular fundamental diagram is

    q(rho) = min{
        v_ff * rho,
        w * (rho_jam - rho)
    }

with

    tau_bar = (1 - pi_CAV) * tau_HDV
              + pi_CAV * tau_CAV

    q_max = v_ff / (v_ff * tau_bar + vehicle_length)

    rho_jam = 1 / vehicle_length

    rho_c = q_max / v_ff

    w = vehicle_length / tau_bar

Outputs:
    - PDF figures for publication
    - 600-dpi PNG figures

The traffic simulation is NOT rerun.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_V_FF_MS = 8.33
DEFAULT_VEHICLE_LENGTH_M = 5.0

DEFAULT_TAU_HDV_S = 3.0
DEFAULT_TAU_CAV_S = 0.9

DEFAULT_TIME_GAP_STUDY_CAV_SHARE = 0.45

DEFAULT_CAV_PENETRATIONS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

DEFAULT_CAV_TIME_GAPS_S = [
    0.6,
    1.2,
    1.8,
    2.4,
    3.0,
]


# ============================================================
# IEEE STYLE
# ============================================================

def set_ieee_style() -> None:
    """
    Apply a consistent IEEE-style plotting format.
    """

    plt.rcParams.update({

        # ----------------------------------------------------
        # Font
        # ----------------------------------------------------
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],

            "mathtext.fontset": "stix",


        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------
        "figure.dpi": 150,
        "savefig.dpi": 600,

        # ----------------------------------------------------
        # Lines / axes
        # ----------------------------------------------------
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.55,

        # ----------------------------------------------------
        # Vector PDF font handling
        # ----------------------------------------------------
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# BRIGHT COLORS
# ============================================================

BRIGHT_COLORS = [
    "#0072BD",   # blue
    "#D95319",   # orange
    "#77AC30",   # green
    "#E6194B",   # red
    "#7E2F8E",   # purple
    "#00A6D6",   # cyan
    "#EDB120",   # gold
    "#FF1493",   # pink
]


# ============================================================
# LINE STYLES
# ============================================================

LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 2)),
    (0, (3, 1, 1, 1)),
]


# ============================================================
# MARKERS
# ============================================================

MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
]


# ============================================================
# COMPOSITION-DEPENDENT FD PARAMETERS
# ============================================================

def composition_weighted_time_gap(
    cav_share: float,
    *,
    tau_hdv_s: float,
    tau_cav_s: float,
) -> float:
    """
    Compute the composition-weighted time gap.

    cav_share:
        Local CAV fraction in [0, 1].
    """

    cav_share = float(cav_share)

    return (
        (1.0 - cav_share) * float(tau_hdv_s)
        + cav_share * float(tau_cav_s)
    )


def fd_parameters(
    *,
    cav_share: float,
    tau_hdv_s: float,
    tau_cav_s: float,
    v_ff_ms: float,
    vehicle_length_m: float,
) -> dict[str, float]:
    """
    Compute MIXSim triangular FD parameters.
    """

    tau_bar_s = composition_weighted_time_gap(
        cav_share,
        tau_hdv_s=tau_hdv_s,
        tau_cav_s=tau_cav_s,
    )

    rho_jam_veh_m = (
        1.0 /
        float(vehicle_length_m)
    )

    q_max_veh_s = (
        float(v_ff_ms) /
        (
            float(v_ff_ms) * tau_bar_s
            + float(vehicle_length_m)
        )
    )

    rho_c_veh_m = (
        q_max_veh_s /
        float(v_ff_ms)
    )

    w_ms = (
        float(vehicle_length_m) /
        tau_bar_s
    )

    return {

        "tau_bar_s":
            tau_bar_s,

        "q_max_veh_s":
            q_max_veh_s,

        "q_max_veh_h":
            3600.0 * q_max_veh_s,

        "rho_c_veh_m":
            rho_c_veh_m,

        "rho_c_veh_km":
            1000.0 * rho_c_veh_m,

        "rho_jam_veh_m":
            rho_jam_veh_m,

        "rho_jam_veh_km":
            1000.0 * rho_jam_veh_m,

        "w_ms":
            w_ms,
    }


# ============================================================
# FUNDAMENTAL DIAGRAM
# ============================================================

def triangular_fd(
    density_veh_km: np.ndarray,
    *,
    cav_share: float,
    tau_hdv_s: float,
    tau_cav_s: float,
    v_ff_ms: float,
    vehicle_length_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Evaluate the composition-dependent triangular FD.
    """

    pars = fd_parameters(

        cav_share=cav_share,

        tau_hdv_s=tau_hdv_s,
        tau_cav_s=tau_cav_s,

        v_ff_ms=v_ff_ms,
        vehicle_length_m=vehicle_length_m,
    )

    density_veh_m = (
        np.asarray(
            density_veh_km,
            dtype=float
        )
        / 1000.0
    )

    free_flow_veh_s = (
        float(v_ff_ms)
        * density_veh_m
    )

    congested_veh_s = (
        pars["w_ms"]
        * np.maximum(
            0.0,
            (
                pars["rho_jam_veh_m"]
                - density_veh_m
            )
        )
    )

    flow_veh_s = np.minimum(
        free_flow_veh_s,
        congested_veh_s
    )

    flow_vph = (
        3600.0 *
        flow_veh_s
    )

    return flow_vph, pars


# ============================================================
# MARKER POSITIONS
# ============================================================

def _markevery(
    n: int,
    n_markers: int = 8
) -> list[int]:

    if n <= 0:
        return []

    return sorted(
        set(
            np.linspace(
                0,
                n - 1,
                min(n_markers, n),
                dtype=int
            ).tolist()
        )
    )


# ============================================================
# IEEE AXIS FORMAT
# ============================================================

def format_ieee_axes(
    ax,
    *,
    x_max: float,
    x_tick: float = 25.0
) -> None:

    ax.grid(False)

    ax.set_xlim(
        0.0,
        x_max
    )

    ax.set_xticks(
        np.arange(
            0.0,
            x_max + 0.1,
            x_tick
        )
    )

    ax.set_ylim(
        bottom=0.0
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=3.0,
        width=0.8,
    )

    for spine in ax.spines.values():

        spine.set_visible(True)
        spine.set_linewidth(0.8)


# ============================================================
# STANDARD IEEE LEGEND
# ============================================================

def format_ieee_legend(
    ax,
    *,
    ncol: int = 3
) -> None:
    """
    Place the legend above the axes so that it does not
    overlap the FD curves.
    """

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=3.0,
        handletextpad=0.6,
    )


# ============================================================
# SAVE FIGURE
# ============================================================

def save_ieee_figure(
    fig,
    output_dir: Path,
    figure_name: str
) -> tuple[Path, Path]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = (
        output_dir /
        f"{figure_name}.pdf"
    )


    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )


    plt.close(fig)

    return pdf_path


# ============================================================
# CAV PENETRATION FIGURE
# ============================================================

def plot_cav_penetration_fd(
    output_dir: Path,
    *,
    cav_penetrations: list[float],
    tau_hdv_s: float,
    tau_cav_s: float,
    v_ff_ms: float,
    vehicle_length_m: float,
) -> tuple[Path, Path]:
    """
    Plot composition-dependent FDs for different CAV shares.

    No numerical annotations are placed inside the figure.
    """

    set_ieee_style()

    rho_jam_veh_km = (
        1000.0 /
        float(vehicle_length_m)
    )

    density_veh_km = np.linspace(
        0.0,
        rho_jam_veh_km,
        1201
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )

    for i, cav_share in enumerate(
        cav_penetrations
    ):

        flow_vph, _ = triangular_fd(

            density_veh_km,

            cav_share=cav_share,

            tau_hdv_s=tau_hdv_s,
            tau_cav_s=tau_cav_s,

            v_ff_ms=v_ff_ms,
            vehicle_length_m=vehicle_length_m,
        )

        color = BRIGHT_COLORS[
            i % len(BRIGHT_COLORS)
        ]

        linestyle = LINESTYLES[
            i % len(LINESTYLES)
        ]

        marker = MARKERS[
            i % len(MARKERS)
        ]

        ax.plot(

            density_veh_km,
            flow_vph,

            color=color,

            linestyle=linestyle,
            linewidth=1.55,

            marker=marker,
            markersize=4.0,

            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,

            markevery=_markevery(
                len(density_veh_km),
                8
            ),

            label=(
                f"{100.0 * cav_share:.0f}% CAV"
            ),
        )

    ax.set_xlabel(
        r"Density [veh/km]"
    )

    ax.set_ylabel(
        r"Flow [veh/h]"
    )

    format_ieee_axes(

        ax,

        x_max=rho_jam_veh_km,

        x_tick=25.0,
    )

    format_ieee_legend(
        ax,
        ncol=3
    )

    fig.tight_layout(
        pad=0.35
    )

    return save_ieee_figure(

        fig,

        output_dir,

        "fig_fd_cav_penetration_ieee"
    )


# ============================================================
# CAV TIME-GAP FIGURE
# ============================================================

def plot_cav_time_gap_fd(
    output_dir: Path,
    *,
    cav_time_gaps_s: list[float],
    cav_share: float,
    tau_hdv_s: float,
    v_ff_ms: float,
    vehicle_length_m: float,
) -> tuple[Path, Path]:
    """
    Plot composition-dependent FDs for different CAV time gaps.

    The CAV penetration remains fixed while tau_CAV changes.
    """

    set_ieee_style()

    rho_jam_veh_km = (
        1000.0 /
        float(vehicle_length_m)
    )

    density_veh_km = np.linspace(
        0.0,
        rho_jam_veh_km,
        1201
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )

    for i, tau_cav_s in enumerate(
        cav_time_gaps_s
    ):

        flow_vph, _ = triangular_fd(

            density_veh_km,

            cav_share=cav_share,

            tau_hdv_s=tau_hdv_s,
            tau_cav_s=tau_cav_s,

            v_ff_ms=v_ff_ms,
            vehicle_length_m=vehicle_length_m,
        )

        color = BRIGHT_COLORS[
            i % len(BRIGHT_COLORS)
        ]

        linestyle = LINESTYLES[
            i % len(LINESTYLES)
        ]

        marker = MARKERS[
            i % len(MARKERS)
        ]

        ax.plot(

            density_veh_km,
            flow_vph,

            color=color,

            linestyle=linestyle,
            linewidth=1.55,

            marker=marker,
            markersize=4.0,

            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,

            markevery=_markevery(
                len(density_veh_km),
                8
            ),

            label=(
                rf"$\tau_{{\mathrm{{CAV}}}}$"
                rf"$={tau_cav_s:.1f}$ s"
            ),
        )

    ax.set_xlabel(
        r"Density [veh/km]"
    )

    ax.set_ylabel(
        r"Flow [veh/h]"
    )

    format_ieee_axes(

        ax,

        x_max=rho_jam_veh_km,

    )

    format_ieee_legend(
        ax,
    )

    fig.tight_layout(
    )

    return save_ieee_figure(

        fig,

        output_dir,

        "fig_fd_cav_time_gap_ieee"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create IEEE-style MIXSim "
            "composition-dependent FD figures "
            "for CAV penetration and time-gap studies."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/fundamental_diagram",
        type=Path
    )

    parser.add_argument(
        "--v-ff",
        type=float,
        default=DEFAULT_V_FF_MS,
        help="Free flow speed [m/s]."
    )

    parser.add_argument(
        "--vehicle-length",
        type=float,
        default=DEFAULT_VEHICLE_LENGTH_M,
        help="Effective standstill spacing [m]."
    )

    parser.add_argument(
        "--tau-hdv",
        type=float,
        default=DEFAULT_TAU_HDV_S,
        help="HDV time gap [s]."
    )

    parser.add_argument(
        "--tau-cav",
        type=float,
        default=DEFAULT_TAU_CAV_S,
        help=(
            "Nominal CAV time gap [s] used "
            "for the CAV-penetration figure."
        )
    )

    parser.add_argument(
        "--time-gap-cav-share",
        type=float,
        default=DEFAULT_TIME_GAP_STUDY_CAV_SHARE,
        help=(
            "Fixed CAV share used in the "
            "CAV time-gap figure."
        )
    )

    parser.add_argument(
        "--cav-penetrations",
        type=float,
        nargs="+",
        default=DEFAULT_CAV_PENETRATIONS,
        help=(
            "CAV shares as fractions, e.g. "
            "0 0.25 0.5 0.75 1."
        )
    )

    parser.add_argument(
        "--cav-time-gaps",
        type=float,
        nargs="+",
        default=DEFAULT_CAV_TIME_GAPS_S,
        help="CAV time gaps [s]."
    )

    args = parser.parse_args()

    out = Path(
        args.output_dir
    )

    fig_dir = (
        out /
        "figures_ieee"
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    penetration_pdf = (
        plot_cav_penetration_fd(

            fig_dir,

            cav_penetrations=list(
                args.cav_penetrations
            ),

            tau_hdv_s=args.tau_hdv,
            tau_cav_s=args.tau_cav,

            v_ff_ms=args.v_ff,
            vehicle_length_m=args.vehicle_length,
        )
    )

    time_gap_pdf = (
        plot_cav_time_gap_fd(

            fig_dir,

            cav_time_gaps_s=list(
                args.cav_time_gaps
            ),

            cav_share=args.time_gap_cav_share,

            tau_hdv_s=args.tau_hdv,

            v_ff_ms=args.v_ff,
            vehicle_length_m=args.vehicle_length,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "IEEE FD figures created successfully"
    )

    print(
        "========================================"
    )

    print(
        "\nFigures:"
    )

    print(
        penetration_pdf
    )


    print(
        time_gap_pdf
    )



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
