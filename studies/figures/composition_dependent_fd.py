#!/usr/bin/env python3
"""
Create an IEEE-style composition-dependent fundamental-diagram
figure for the MIXSim modeling framework.

The figure shows:

    - Godunov demand function D(rho)
    - Godunov supply function S(rho)
    - resulting triangular fundamental diagram

        q(rho) = min{D(rho), S(rho)}

For the MIXSim composition-dependent formulation:

    tau_bar = (1 - pi_CAV) * tau_HDV
              + pi_CAV * tau_CAV

    q_max = v_ff / (v_ff * tau_bar + vehicle_length)

    rho_jam = 1 / vehicle_length

    rho_c = q_max / v_ff

    w = vehicle_length / tau_bar

and

    D(rho) = min{
        v_ff * rho,
        q_max
    }

    S(rho) = min{
        q_max,
        w * (rho_jam - rho)
    }

A representative local composition is used only to draw the
schematic curve. Numerical parameter values are intentionally
not annotated inside the figure.

Outputs:
    - PDF figure for publication
    - 600-dpi PNG figure

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

DEFAULT_CAV_SHARE = 0.50


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
    "--",
    ":",
    "-",
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
    """

    return (
        (1.0 - float(cav_share))
        * float(tau_hdv_s)
        +
        float(cav_share)
        * float(tau_cav_s)
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
    Compute composition-dependent triangular FD parameters.
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
# DEMAND / SUPPLY / FUNDAMENTAL DIAGRAM
# ============================================================

def demand_supply_fd(
    density_veh_km: np.ndarray,
    *,
    cav_share: float,
    tau_hdv_s: float,
    tau_cav_s: float,
    v_ff_ms: float,
    vehicle_length_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    """
    Evaluate the Godunov demand, supply, and resulting
    triangular fundamental diagram.
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

    demand_veh_s = np.minimum(

        float(v_ff_ms)
        * density_veh_m,

        pars["q_max_veh_s"],
    )

    supply_veh_s = np.minimum(

        pars["q_max_veh_s"],

        pars["w_ms"]
        * np.maximum(
            0.0,
            (
                pars["rho_jam_veh_m"]
                - density_veh_m
            )
        ),
    )

    flow_veh_s = np.minimum(
        demand_veh_s,
        supply_veh_s
    )

    return (
        3600.0 * demand_veh_s,
        3600.0 * supply_veh_s,
        3600.0 * flow_veh_s,
        pars,
    )


# ============================================================
# IEEE AXIS FORMAT
# ============================================================

def format_ieee_axes(
    ax,
    *,
    x_max: float,
    x_tick: float = 25.0,
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
    ax
) -> None:
    """
    Keep the legend above the figure so that it does not
    cover the demand, supply, or FD curves.
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

    png_path = (
        output_dir /
        f"{figure_name}.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=600
    )

    plt.close(fig)

    return pdf_path, png_path


# ============================================================
# COMPOSITION-DEPENDENT FD FIGURE
# ============================================================

def plot_composition_dependent_fd(
    output_dir: Path,
    *,
    cav_share: float,
    tau_hdv_s: float,
    tau_cav_s: float,
    v_ff_ms: float,
    vehicle_length_m: float,
) -> tuple[Path, Path]:
    """
    Plot the Godunov demand and supply functions together
    with the resulting composition-dependent triangular FD.

    Numerical parameter values are not annotated inside
    the figure.
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

    (
        demand_vph,
        supply_vph,
        flow_vph,
        pars,
    ) = demand_supply_fd(

        density_veh_km,

        cav_share=cav_share,

        tau_hdv_s=tau_hdv_s,
        tau_cav_s=tau_cav_s,

        v_ff_ms=v_ff_ms,
        vehicle_length_m=vehicle_length_m,
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )

    ax.plot(

        density_veh_km,
        demand_vph,

        color=BRIGHT_COLORS[0],

        linestyle=LINESTYLES[0],
        linewidth=1.55,

        label=r"Demand $D(\rho)$",
    )

    ax.plot(

        density_veh_km,
        supply_vph,

        color=BRIGHT_COLORS[1],

        linestyle=LINESTYLES[1],
        linewidth=1.70,

        label=r"Supply $S(\rho)$",
    )

    ax.plot(

        density_veh_km,
        flow_vph,

        color=BRIGHT_COLORS[2],

        linestyle=LINESTYLES[2],
        linewidth=2.0,

        label=(
            r"$q(\rho)=\min\{D(\rho),S(\rho)\}$"
        ),
    )

    ax.plot(

        pars["rho_c_veh_km"],
        pars["q_max_veh_h"],

        color="black",

        marker="o",
        markersize=5.0,

        linestyle="None",
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

    ax.set_ylim(
        0.0,
        1.08 * pars["q_max_veh_h"]
    )

    format_ieee_legend(
        ax
    )

    fig.tight_layout(
        pad=0.35
    )

    return save_ieee_figure(

        fig,

        output_dir,

        "fig_composition_dependent_fd_ieee"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create the IEEE-style MIXSim "
            "composition-dependent demand, supply, "
            "and triangular FD figure."
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
        help="CAV time gap [s]."
    )

    parser.add_argument(
        "--cav-share",
        type=float,
        default=DEFAULT_CAV_SHARE,
        help=(
            "Representative local CAV share "
            "used to draw the composition-dependent FD."
        )
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

    pdf_path, _ = (
        plot_composition_dependent_fd(

            fig_dir,

            cav_share=args.cav_share,

            tau_hdv_s=args.tau_hdv,
            tau_cav_s=args.tau_cav,

            v_ff_ms=args.v_ff,
            vehicle_length_m=args.vehicle_length,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "IEEE composition-dependent FD figure "
        "created successfully"
    )

    print(
        "========================================"
    )

    print(
        "\nFigures:"
    )

    print(
        pdf_path
    )




# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
