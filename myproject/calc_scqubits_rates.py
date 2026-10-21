import os
import sys
import numpy as np
import scipy.constants as const

# Ensure repo root and script directory are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import scqubits as scq
import fluxonium_bpf as bpf

# Disable scqubits default t1 warning for clean output
scq.settings.T1_DEFAULT_WARNING = False


def run_scqubits_rates():
    EJ = 5.3    # Josephson energy in GHz
    EC = 1.36   # Charging energy in GHz
    EL = 0.146  # Inductive energy in GHz
    flux = 0.0  # External flux bias in Phi_0
    cutoff = 110

    # Build Fluxonium qubit
    fl = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=flux, cutoff=cutoff)

    # Calculate C_sigma total qubit capacitance
    EC_J = EC * 1e9 * const.h
    C_sigma = (const.e**2) / (2.0 * EC_J)

    env_params = {
        'C_sigma': C_sigma,
        'Q_cap': 1e6,
        'Cc': 5e-15,
        'Z0': 50.0,
        'f_filter_hz': 5.9e9,
        'bw_hz': 500e6,
        'N_order': 6,
        'g_prods': [0.518, 0.732, 2.732, 3.732, 2.732, 0.732, 0.518],
        'coupling_stage': 3
    }

    # Z for transmission line (scalar impedance)
    Z_tl = (env_params['Cc'] / env_params['C_sigma'])**2 * env_params['Z0']

    # Z for multi-stage BPF (callable impedance vs omega)
    # Note: scqubits passes omega in system angular frequency units (2*pi*GHz).
    # We convert omega to rad/s by multiplying by 1e9 before passing to compute_environmental_impedance.
    def Z_bpf(omega_sys):
        omega_rad_s = omega_sys * 1e9
        return bpf.compute_environmental_impedance(omega_rad_s, 'multi_stage_bpf', env_params)

    environments = [
        ('transmission_line', 'Transmission Line', Z_tl),
        ('multi_stage_bpf', 'Multi-Stage BPF', Z_bpf)
    ]
    temperatures = [
        (0.0, '0K'),
        (0.070, '70mK')
    ]

    print("\n" + "=" * 70)
    print("SCQUBITS T1_CHARGE_IMPEDANCE TRANSITION RATES (0<->3)")
    print("=" * 70)

    results = {}
    for env_key, env_name, Z_val in environments:
        results[env_key] = {}
        for T_k, T_str in temperatures:
            # scqubits t1_charge_impedance with total=False returns single-directional transition rate.
            # get_rate=True returns rate in system frequency units (GHz).
            # Convert GHz -> MHz by multiplying by 1e3.
            rate_03_ghz = fl.t1_charge_impedance(i=0, j=3, Z=Z_val, T=T_k, total=False, get_rate=True)
            rate_30_ghz = fl.t1_charge_impedance(i=3, j=0, Z=Z_val, T=T_k, total=False, get_rate=True)

            rate_03_mhz = rate_03_ghz * 1e3
            rate_30_mhz = rate_30_ghz * 1e3

            results[env_key][T_str] = {'0->3': rate_03_mhz, '3->0': rate_30_mhz}

            print(f"\nEnvironment: {env_name} | Temperature: {T_str}")
            print(f"  Rate 0->3 (Excitation): {rate_03_mhz:.10e} MHz")
            print(f"  Rate 3->0 (Decay)     : {rate_30_mhz:.10e} MHz")

    return results


if __name__ == '__main__':
    run_scqubits_rates()
