"""Replay STIRAP pulse from CSV with customizable environment and temperature.

Supports two pulse loading modes:
1. Direct CSV replay: Reads Omega_p and Omega_s directly from the CSV columns.
2. Coefficient-based replay: Reads Omega(t) and time grid from CSV, and reconstructs
   theta(t) and the pulse envelopes using manually provided Legendre coefficients:
       Omega_p(t) = Omega(t) * sin(theta(t))
       Omega_s(t) = Omega(t) * cos(theta(t))

Usage:
    # Mode 1: Direct replay from CSV
    python replay_stirap_pulse.py --csv optimization_history/pulse_iter_000.csv --env multi_stage_bpf --temp 0.030

    # Mode 2: Reconstruct using copied coefficients
    python replay_stirap_pulse.py --csv optimization_history/pulse_iter_000.csv --coeffs "-0.00633, 0.0496, 0.0, 0.0, 0.0, 0.0" --env multi_stage_bpf --temp 0.030
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fluxonium_bpf as model
from weighted_legendre_pulse_optimization import (
    simulate_stirap_arbitrary_pulse,
    plot_stirap_results,
    build_weighted_legendre_basis,
    reconstruct_eta_theta,
)


def load_pulse_csv(
    csv_path: str,
    coefficients: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load time and Rabi frequencies (in rad/ns) from a saved pulse CSV file.

    If `coefficients` is provided, Omega(t) is extracted from the CSV and
    theta(t) is reconstructed using the weighted Legendre basis:
        Omega_p(t) = Omega(t) * sin(theta(t))
        Omega_s(t) = Omega(t) * cos(theta(t))

    Returns:
        (t_grid, Omega_p_rad_ns, Omega_s_rad_ns, Omega_rad_ns, theta_rad)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Pulse CSV file not found: {csv_path}")

    # Read header to find column indices
    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    col_names = [c.strip() for c in header_line.split(",")]
    col_map = {name: idx for idx, name in enumerate(col_names)}

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    # Extract time grid
    if "time_ns" in col_map:
        t_grid = data[:, col_map["time_ns"]]
    else:
        t_grid = data[:, 0]

    # Extract base envelope Omega
    if "Omega_rad_ns" in col_map:
        omega_base = data[:, col_map["Omega_rad_ns"]]
    elif "Omega_mhz" in col_map:
        omega_base = data[:, col_map["Omega_mhz"]] * (2.0 * np.pi * 1e-3)
    else:
        omega_base = None

    # If coefficients are provided, reconstruct theta and pulse envelopes
    if coefficients is not None and len(coefficients) > 0:
        if omega_base is None:
            raise ValueError("CSV must contain 'Omega_rad_ns' or 'Omega_mhz' to reconstruct from coefficients.")
        coeffs = np.asarray(coefficients, dtype=float)
        max_deg = len(coeffs)
        basis = build_weighted_legendre_basis(t_grid, omega_base, max_deg)
        _, theta = reconstruct_eta_theta(t_grid, omega_base, basis, coeffs)
        omega_p = omega_base * np.sin(theta)
        omega_s = omega_base * np.cos(theta)
        return t_grid, omega_p, omega_s, omega_base, theta

    # Otherwise read existing Omega_p and Omega_s from CSV
    if "Omega_p_rad_ns" in col_map:
        omega_p = data[:, col_map["Omega_p_rad_ns"]]
    elif "Omega_p_mhz" in col_map:
        omega_p = data[:, col_map["Omega_p_mhz"]] * (2.0 * np.pi * 1e-3)
    else:
        raise KeyError("Could not find 'Omega_p_rad_ns' or 'Omega_p_mhz' column in CSV.")

    if "Omega_s_rad_ns" in col_map:
        omega_s = data[:, col_map["Omega_s_rad_ns"]]
    elif "Omega_s_mhz" in col_map:
        omega_s = data[:, col_map["Omega_s_mhz"]] * (2.0 * np.pi * 1e-3)
    else:
        raise KeyError("Could not find 'Omega_s_rad_ns' or 'Omega_s_mhz' column in CSV.")

    theta = data[:, col_map["theta_rad"]] if "theta_rad" in col_map else np.arctan2(omega_p, omega_s)
    if omega_base is None:
        omega_base = np.sqrt(omega_p**2 + omega_s**2)

    return t_grid, omega_p, omega_s, omega_base, theta


def build_system_cache(
    env_type: str = "multi_stage_bpf",
    T_kelvin: float = 0.030,
) -> dict[str, Any]:
    """Construct Fluxonium eigen-system, dissipation rates, and collapse operators.

    Parameters:
        env_type: 'multi_stage_bpf', 'transmission_line', or 'uncoupled'
        T_kelvin: Temperature in Kelvin (e.g. 0.030 for 30 mK)
    """
    system_data = model.setup_fluxonium_system()
    N = system_data["N_LEVELS"]
    if N < 4:
        raise ValueError("0-3-2 STIRAP requires at least 4 energy levels.")

    EC_J = system_data["EC"] * 1e9 * model.const.h
    C_sigma = model.const.e**2 / (2.0 * EC_J)

    env_params = {
        "C_sigma": C_sigma,
        "Q_cap": 1e6,
        "Cc": 1.1e-15,
        "Z0": 50.0,
        "f_filter_hz": 5.8e9,
        "bw_hz": 1000e6,
        "N_order": 6,
        "g_prods": [0.518, 0.732, 2.732, 3.732, 2.732, 0.732, 0.518],
        "coupling_stage": 3,
    }

    rates = model.calculate_transition_rates(
        system_data=system_data,
        T_kelvin=T_kelvin,
        env_type=env_type,
        params=env_params,
    )

    c_ops = []
    for item in rates:
        initial = int(item["from"])
        final = int(item["to"])
        gamma_ns = float(item["rate_mhz"]) * 1e-3
        if gamma_ns > 0.0:
            jump_op = qt.basis(N, final) * qt.basis(N, initial).dag()
            c_ops.append(np.sqrt(gamma_ns) * jump_op)

    H0 = 2.0 * np.pi * system_data["H_qutip"]
    n_op = system_data["n_qutip"]
    n_array = n_op.full()
    E = np.asarray(system_data["evals_ghz"], dtype=float)

    omega_p = 2.0 * np.pi * (E[3] - E[0])  # Pump: 0 <-> 3
    omega_s = 2.0 * np.pi * (E[3] - E[2])  # Stokes: 2 <-> 3

    n03 = abs(n_array[3, 0])
    n32 = abs(n_array[3, 2])
    if n03 < 1e-14 or n32 < 1e-14:
        raise ValueError("Matrix elements n03 or n32 are approximately zero.")

    return {
        "system_data": system_data,
        "N": N,
        "H0": H0,
        "n_op": n_op,
        "n_array": n_array,
        "E": E,
        "omega_p": omega_p,
        "omega_s": omega_s,
        "n03": n03,
        "n32": n32,
        "rates": rates,
        "c_ops": c_ops,
        "env_type": env_type,
        "T_kelvin": T_kelvin,
    }


def replay_stirap_simulation(
    csv_path: str,
    env_type: str = "multi_stage_bpf",
    T_kelvin: float = 0.030,
    coefficients: Optional[np.ndarray | list[float] | str] = None,
    output_prefix: Optional[str] = None,
) -> dict[str, Any]:
    """Run STIRAP simulation with loaded CSV pulse envelopes under chosen environment & temperature.

    Parameters:
        csv_path: Path to pulse CSV file.
        env_type: 'multi_stage_bpf', 'transmission_line', or 'uncoupled'.
        T_kelvin: Temperature in Kelvin (e.g. 0.030 for 30 mK).
        coefficients: Optional Legendre coefficients (array, list, or comma-separated string).
                      If supplied, theta(t) and pulse envelopes are reconstructed from Omega(t).
        output_prefix: Output file prefix for generated figures and text reports.
    """
    if isinstance(coefficients, str):
        # Parse string like "-0.00633, 0.0496, 0.0"
        coeff_list = [float(x.strip()) for x in coefficients.split(",") if x.strip()]
        coeffs_arr = np.asarray(coeff_list, dtype=float)
    elif coefficients is not None:
        coeffs_arr = np.asarray(coefficients, dtype=float)
    else:
        coeffs_arr = None

    print("=" * 70)
    print("REPLAY STIRAP PULSE SIMULATION")
    print("=" * 70)
    print(f"Loading pulse from: {csv_path}")
    if coeffs_arr is not None:
        print(f"Reconstructing pulse using {len(coeffs_arr)} coefficients: {np.round(coeffs_arr, 6)}")
    else:
        print("Using pulse envelopes directly from CSV columns.")
    print(f"Target Environment: {env_type}")
    print(f"Target Temperature: {T_kelvin * 1e3:.1f} mK ({T_kelvin} K)")

    # 1. Load / reconstruct pulse envelopes
    t_grid, omega_p, omega_s, omega_base, theta = load_pulse_csv(csv_path, coefficients=coeffs_arr)
    print(f"Pulse ready: {len(t_grid)} points from {t_grid[0]:.2f} ns to {t_grid[-1]:.2f} ns.")

    # 2. Build system with new environment and temperature
    system_cache = build_system_cache(env_type=env_type, T_kelvin=T_kelvin)
    print(f"System initialized with {len(system_cache['c_ops'])} collapse operators.")

    # 3. Simulate population dynamics
    print("\nRunning QuTiP mesolve dynamics...")
    N = system_cache["N"]
    P0 = qt.basis(N, 0).proj()
    P2 = qt.basis(N, 2).proj()
    P3 = qt.basis(N, 3).proj()
    P_other = qt.qeye(N) - P0 - P2 - P3

    sim_res = simulate_stirap_arbitrary_pulse(
        t_grid=t_grid,
        Omega_p=omega_p,
        Omega_s=omega_s,
        system_cache=system_cache,
        e_ops=[P0, P3, P2, P_other],
    )

    P0_t, P3_t, P2_t, Pother_t = [np.real(np.asarray(x)) for x in sim_res.expect]

    final_P0 = float(P0_t[-1])
    final_P2 = float(P2_t[-1])
    final_P3 = float(P3_t[-1])
    final_Pother = float(Pother_t[-1])
    p3_max = float(np.max(P3_t))

    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"Final Fidelity (P2) = {final_P2:.6f}")
    print(f"Final P0            = {final_P0:.6f}")
    print(f"Final P3            = {final_P3:.6f}")
    print(f"Final P_other       = {final_Pother:.6f}")
    print(f"Maximum P3          = {p3_max:.6f}")

    # 4. Save results text and plots
    temp_str = f"{int(round(T_kelvin * 1e3))}mK"
    if output_prefix is None:
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_prefix = f"replay_{csv_basename}_{env_type}_{temp_str}"

    pulse_fig = f"{output_prefix}_pulses.png"
    pop_fig = f"{output_prefix}_populations.png"
    txt_file = f"{output_prefix}_results.txt"

    plot_stirap_results(
        tlist=t_grid,
        Omega_p=omega_p,
        Omega_s=omega_s,
        P0_t=P0_t,
        P3_t=P3_t,
        P2_t=P2_t,
        Pother_t=Pother_t,
        pulse_filename=pulse_fig,
        pop_filename=pop_fig,
    )

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("STIRAP PULSE REPLAY RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source CSV:    {csv_path}\n")
        if coeffs_arr is not None:
            f.write(f"Coefficients:  {np.array2string(coeffs_arr, precision=8, separator=', ')}\n")
        f.write(f"Environment:   {env_type}\n")
        f.write(f"Temperature:   {T_kelvin * 1e3:.1f} mK ({T_kelvin} K)\n")
        f.write(f"Fidelity (P2): {final_P2:.8f}\n")
        f.write(f"Final P0:      {final_P0:.8f}\n")
        f.write(f"Final P2:      {final_P2:.8f}\n")
        f.write(f"Final P3:      {final_P3:.8f}\n")
        f.write(f"Final P_other: {final_Pother:.8f}\n")
        f.write(f"P3_max:        {p3_max:.8f}\n")
        f.write("=" * 60 + "\n")

    print(f"Saved results summary to: {txt_file}")

    return {
        "t_grid": t_grid,
        "Omega_p": omega_p,
        "Omega_s": omega_s,
        "P0_t": P0_t,
        "P2_t": P2_t,
        "P3_t": P3_t,
        "Pother_t": Pother_t,
        "fidelity": final_P2,
        "final_P0": final_P0,
        "final_P3": final_P3,
        "final_Pother": final_Pother,
        "p3_max": p3_max,
        "pulse_fig": pulse_fig,
        "pop_fig": pop_fig,
        "txt_file": txt_file,
    }


def find_default_pulse_csv() -> str:
    """Find a default pulse CSV in optimization_history or current directory."""
    candidate_dirs = ["optimization_history", "."]
    for c_dir in candidate_dirs:
        if os.path.exists(c_dir):
            csvs = sorted([f for f in os.listdir(c_dir) if f.startswith("pulse_iter_") and f.endswith(".csv")])
            if csvs:
                # Return the latest iteration
                return os.path.join(c_dir, csvs[-1])
    return "optimization_history/pulse_iter_000.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay STIRAP pulse from CSV with customizable environment and temperature."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to pulse CSV file (default: latest pulse_iter_*.csv in optimization_history/)",
    )
    parser.add_argument(
        "--coeffs",
        type=str,
        default=None,
        help="Legendre pulse coefficients as comma-separated floats, e.g. '-0.00633, 0.0496, 0.0' (optional). "
             "If omitted, uses Omega_p and Omega_s directly from CSV.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="multi_stage_bpf",
        choices=["transmission_line", "multi_stage_bpf", "uncoupled"],
        help="Environmental coupling type: 'multi_stage_bpf', 'transmission_line', or 'uncoupled' (default: multi_stage_bpf)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.030,
        help="Temperature in Kelvin, e.g. 0.030 for 30 mK, 0.0 for 0 K (default: 0.030)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Prefix for output figures and result text file (default: auto-generated)",
    )

    args = parser.parse_args()

    csv_file = args.csv if args.csv is not None else find_default_pulse_csv()
    replay_stirap_simulation(
        csv_path=csv_file,
        env_type=args.env,
        T_kelvin=args.temp,
        coefficients=args.coeffs,
        output_prefix=args.out,
    )
