import os
import sys
import scipy.constants as const

# Ensure parent / script directory is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import fluxonium_bpf as bpf


def run_bpf_rates():
    # 1. Setup system
    system_data = bpf.setup_fluxonium_system()

    # 2. Setup parameters
    EC_J = system_data['EC'] * 1e9 * const.h
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

    environments = [
        ('transmission_line', 'Transmission Line'),
        ('multi_stage_bpf', 'Multi-Stage BPF')
    ]
    temperatures = [
        (0.0, '0K'),
        (0.070, '70mK')
    ]

    print("\n" + "=" * 70)
    print("FLUXONIUM_BPF.PY TRANSITION RATES (0<->3)")
    print("=" * 70)

    results = {}
    for env_key, env_name in environments:
        results[env_key] = {}
        for T_k, T_str in temperatures:
            rates = bpf.calculate_transition_rates(system_data, T_k, env_key, env_params)

            # Extract 0->3 (excitation) and 3->0 (decay)
            rate_03 = next((t['rate_mhz'] for t in rates if t['label'] == '0->3'), 0.0)
            rate_30 = next((t['rate_mhz'] for t in rates if t['label'] == '3->0'), 0.0)

            results[env_key][T_str] = {'0->3': rate_03, '3->0': rate_30}

            print(f"\nEnvironment: {env_name} | Temperature: {T_str}")
            print(f"  Rate 0->3 (Excitation): {rate_03:.10e} MHz")
            print(f"  Rate 3->0 (Decay)     : {rate_30:.10e} MHz")

    return results


if __name__ == '__main__':
    run_bpf_rates()
