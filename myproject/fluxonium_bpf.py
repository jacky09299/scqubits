import os
import sys

# Ensure installed site-packages qutip is imported if running from inside qutip source directory
repo_dir = os.path.abspath(os.path.dirname(__file__))
if repo_dir in sys.path:
    sys.path.remove(repo_dir)
if '' in sys.path:
    sys.path.remove('')

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import qutip as qt
import scqubits as scq

# Re-add script directory to sys.path for standard imports
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# Set plot style for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})


def setup_fluxonium_system():
    """
    Constructs the Fluxonium qubit using scqubits and extracts system properties
    (eigenenergies, number operator matrix elements, and qutip representation).
    """
    EJ = 5.3    # Josephson energy in GHz
    EC = 1.36   # Charging energy in GHz
    EL = 0.146  # Inductive energy in GHz
    flux = 0.0  # External flux bias in units of Phi_0
    cutoff = 110 # Harmonic oscillator cutoff basis size
    N_LEVELS = 6 # Number of energy levels to extract

    print("=" * 70)
    print("FLUXONIUM QUBIT PARAMETERS")
    print("=" * 70)
    print(f"EJ = {EJ} GHz | EC = {EC} GHz | EL = {EL} GHz | flux = {flux} Phi_0")
    print(f"Cutoff basis size = {cutoff} | N_LEVELS = {N_LEVELS}")

    # Build scqubits Fluxonium object
    fluxonium = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=flux, cutoff=cutoff)
    evals, evecs = fluxonium.eigensys(evals_count=N_LEVELS)

    # Calculate transition energies (relative to ground state E0)
    evals_rel_ghz = evals - evals[0]
    print("\nCalculated Eigenenergies (relative to E0):")
    for idx, E in enumerate(evals_rel_ghz):
        print(f"  E_{idx} = {E:.4f} GHz")

    # Extract matrix elements of charge operator n = Q / (2e)
    n_mat = fluxonium.matrixelement_table('n_operator', evals_count=N_LEVELS)

    # QuTiP representation of Hamiltonian & n_operator in qubit eigenbasis
    H_qutip = qt.Qobj(np.diag(evals_rel_ghz))
    n_qutip = qt.Qobj(n_mat)

    return {
        'EJ': EJ,
        'EC': EC,
        'EL': EL,
        'N_LEVELS': N_LEVELS,
        'evals_ghz': evals_rel_ghz,
        'n_mat': n_mat,
        'H_qutip': H_qutip,
        'n_qutip': n_qutip
    }


def compute_environmental_impedance(w_rad_s, env_type, params):
    """
    Computes real part of environmental impedance Re[Z_env(w)] in Ohms.
    """
    abs_w = np.abs(w_rad_s)
    if abs_w < 1e-6:
        return 0.0

    if env_type == 'uncoupled':
        # Re[Z(w)] = 1 / (w * C_sigma * Q_cap)
        C_sigma = params['C_sigma']
        Q_cap = params['Q_cap']
        return 1.0 / (abs_w * C_sigma * Q_cap)

    elif env_type == 'transmission_line':
        Cc, Z0, C_sigma = params['Cc'], params['Z0'], params['C_sigma']
        return (Cc / C_sigma) ** 2 * Z0
        
    elif env_type == 'multi_stage_bpf':
        Cc = params['Cc']
        Z0 = params['Z0']
        w_0 = 2.0 * np.pi * params['f_filter_hz']
        dw = 2.0 * np.pi * params['bw_hz']
        
        N = params['N_order']
        g_prods = params['g_prods']
        k_stage = params['coupling_stage'] - 1 # 轉為 0-indexed
        
        # 1. 計算各級耗散率 (kappas) 與級間耦合強度 (c)
        kappas = np.zeros(N)
        kappas[0] = dw / g_prods[0]   # 第一級與環境的耦合 
        kappas[-1] = dw / g_prods[-1] # 最後一級與環境的耦合 
        
        c = np.zeros(N-1)
        for j in range(N-1):
            c[j] = dw / (2.0 * np.sqrt(g_prods[j+1])) 
            
        # 2. 建構 N x N 的耦合模式矩陣 M 
        M = np.zeros((N, N), dtype=complex)
        for j in range(N):
            M[j, j] = -1j * w_0 - kappas[j] / 2.0 
            if j < N - 1:
                M[j, j+1] = -1j * c[j] 
                M[j+1, j] = -1j * c[j] 

        # 3. 計算指定躍遷頻率 abs_w 下的格林函數 (i*w*I + M)^-1 
        I_mat = np.eye(N)
        matrix_w = 1j * abs_w * I_mat + M
        inv_matrix_w = np.linalg.inv(matrix_w)
        
        # 為了將 LDOS 對應回古典 50 Ohm 阻抗，同時計算中心頻率 w_0 的響應
        matrix_0 = 1j * w_0 * I_mat + M
        inv_matrix_0 = np.linalg.inv(matrix_0)
        
        # 4. 從反矩陣的對角線元素提取局部狀態密度 (LDOS) 
        rho_k_w = - (1.0 / np.pi) * np.real(inv_matrix_w[k_stage, k_stage]) 
        rho_k_0 = - (1.0 / np.pi) * np.real(inv_matrix_0[k_stage, k_stage])
        
        # 5. 阻抗與 LDOS 成正比，以中心頻率的 Z0 進行正規化縮放 
        Z_bpf_w = Z0 * (rho_k_w / rho_k_0)

        # 考慮耦合電容 Cc 後的等效阻抗
        return (Cc / params['C_sigma']) ** 2 * Z_bpf_w

    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def compute_noise_psd(w_rad_s, T_kelvin, Re_Z):
    """
    Computes S_VV(w) voltage noise spectral density in V^2 / Hz.
    S_VV(w) = 2 * hbar * w / (1 - exp(-hbar * w / (kB * T))) * Re[Z(w)]
    """
    if Re_Z <= 0.0:
        return 0.0

    kB = const.k
    hbar = const.hbar

    if T_kelvin == 0.0:
        if w_rad_s > 0.0: # Emission (transition from higher to lower energy level)
            return 2.0 * hbar * w_rad_s * Re_Z
        else: # Absorption is impossible at T = 0 K
            return 0.0
    else:
        exponent = (hbar * w_rad_s) / (kB * T_kelvin)
        if exponent < -700.0:
            return 0.0
        elif exponent > 700.0:
            return 2.0 * hbar * w_rad_s * Re_Z
        else: # Standard thermal formula
            return (2.0 * hbar * w_rad_s / (1.0 - np.exp(-exponent))) * Re_Z


def calculate_transition_rates(system_data, T_kelvin, env_type, params):
    """
    Calculates transition rates Gamma_{j -> i} for allowed level pairs (j != i and |j - i| is odd).
    Forbidden transitions (|j - i| is even) are excluded. Rate values are in MHz.
    """
    evals_ghz = system_data['evals_ghz']
    n_mat = system_data['n_mat']
    N_LEVELS = system_data['N_LEVELS']
    e = const.e
    hbar = const.hbar

    transitions = []

    for j in range(N_LEVELS): # Initial state j
        for i in range(N_LEVELS): # Final state i
            if i == j:
                continue

            # Filter forbidden transitions (even energy level difference |j - i|)
            if abs(j - i) % 2 == 0:
                continue

            # Energy difference E_j - E_i (GHz)
            delta_E_ghz = evals_ghz[j] - evals_ghz[i]
            w_ij = 2.0 * np.pi * delta_E_ghz * 1e9 # rad/s

            abs_delta_E_ghz = np.abs(delta_E_ghz)
            if abs_delta_E_ghz < 1e-6:
                continue

            # Environmental impedance Re[Z(w)]
            Re_Z = compute_environmental_impedance(w_ij, env_type, params)

            # Noise Spectral Density S_VV(w)
            S_vv = compute_noise_psd(w_ij, T_kelvin, Re_Z)

            # Matrix element squared |<i|n|j>|^2
            n_element_sq = np.abs(n_mat[i, j])**2

            # Fermi's Golden Rule rate Gamma_{j -> i}
            # Gamma = (2e / hbar)^2 * |n_ij|^2 * S_VV(w)
            rate_hz = ((2.0 * e / hbar)**2) * n_element_sq * S_vv
            rate_mhz = rate_hz / 1e6 # Convert rate to MHz

            if rate_mhz > 0:
                transitions.append({
                    'from': j,
                    'to': i,
                    'label': f"{j}->{i}",
                    'transition_type': 'decay' if j > i else 'excitation',
                    'energy_ghz': abs_delta_E_ghz,
                    'rate_mhz': rate_mhz
                })

    return transitions


def apply_unified_log_axes(ax, ylim):
    """
    Applies unified log y-scale, decade major grid lines, and lighter minor grid lines.
    """
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))

    ax.grid(True, which="major", color="gray", linestyle="-", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="minor", color="gray", linestyle=":", linewidth=0.4, alpha=0.4)


def create_base_plot(transitions, env_name, temp_str, filename, ylim=None):
    """
    Generates one of the 6 individual transition rate vs transition frequency plots.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Split into decay (j > i) and excitation (j < i)
    decays = [t for t in transitions if t['transition_type'] == 'decay']
    excitations = [t for t in transitions if t['transition_type'] == 'excitation']

    if decays:
        x_d = [t['energy_ghz'] for t in decays]
        y_d = [t['rate_mhz'] for t in decays]
        ax.scatter(x_d, y_d, color='blue', s=70, zorder=4, label='Decay ($j \\to i$, $j > i$)')

        for t in decays:
            ax.annotate(t['label'], (t['energy_ghz'], t['rate_mhz']),
                        textcoords="offset points", xytext=(5, 5), ha='left',
                        fontsize=9, color='darkblue', fontweight='bold')

    if excitations:
        x_e = [t['energy_ghz'] for t in excitations]
        y_e = [t['rate_mhz'] for t in excitations]
        ax.scatter(x_e, y_e, color='red', marker='^', s=70, zorder=4, label='Excitation ($j \\to i$, $j < i$)')

        for t in excitations:
            ax.annotate(t['label'], (t['energy_ghz'], t['rate_mhz']),
                        textcoords="offset points", xytext=(5, -12), ha='left',
                        fontsize=9, color='darkred', fontweight='bold')

    apply_unified_log_axes(ax, ylim)
    ax.set_xlabel('Transition Energy $\\hbar\\omega_{ij} / h$ (GHz)')
    ax.set_ylabel('Transition Rate $\\Gamma_{j \\to i}$ (MHz)')
    ax.set_title(f'Fluxonium Transition Rates ({env_name}, T = {temp_str})')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot: {filename}")


def create_temp_overlay_plot(trans_0K, trans_70mK, env_name, filename, ylim=None):
    """
    Generates overlay plot comparing T = 0 K vs T = 70 mK for a given environment.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))

    x_0 = [t['energy_ghz'] for t in trans_0K]
    y_0 = [t['rate_mhz'] for t in trans_0K]
    ax.scatter(x_0, y_0, color='blue', s=80, marker='o', alpha=0.8, zorder=4, label='T = 0 K (Decay only)')

    x_70 = [t['energy_ghz'] for t in trans_70mK]
    y_70 = [t['rate_mhz'] for t in trans_70mK]
    ax.scatter(x_70, y_70, color='crimson', s=60, marker='x', zorder=5, label='T = 70 mK (Decay + Excitation)')

    for t in trans_70mK:
        color = 'darkred' if t['transition_type'] == 'excitation' else 'darkblue'
        offset = (5, -10) if t['transition_type'] == 'excitation' else (5, 5)
        ax.annotate(t['label'], (t['energy_ghz'], t['rate_mhz']),
                    textcoords="offset points", xytext=offset, ha='left',
                    fontsize=8.5, color=color)

    apply_unified_log_axes(ax, ylim)
    ax.set_xlabel('Transition Energy $\\hbar\\omega_{ij} / h$ (GHz)')
    ax.set_ylabel('Transition Rate $\\Gamma_{j \\to i}$ (MHz)')
    ax.set_title(f'Temperature Overlay: 0 K vs 70 mK ({env_name})')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved overlay plot: {filename}")


def create_env_overlay_plot(trans_TL, trans_BPF, temp_str, filename, ylim=None):
    """
    Generates overlay plot comparing Transmission Line vs Bandpass Filter for a given temperature.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))

    x_tl = [t['energy_ghz'] for t in trans_TL]
    y_tl = [t['rate_mhz'] for t in trans_TL]
    ax.scatter(x_tl, y_tl, color='teal', s=80, marker='s', alpha=0.8, zorder=4, label='Transmission Line ($Z_0 = 50\\,\\Omega$)')

    x_bpf = [t['energy_ghz'] for t in trans_BPF]
    y_bpf = [t['rate_mhz'] for t in trans_BPF]
    ax.scatter(x_bpf, y_bpf, color='magenta', s=80, marker='D', alpha=0.8, zorder=5, label='Bandpass Filter (Purcell Protected)')

    for t in trans_BPF:
        ax.annotate(t['label'], (t['energy_ghz'], t['rate_mhz']),
                    textcoords="offset points", xytext=(5, 5), ha='left',
                    fontsize=8.5, color='darkmagenta')

    apply_unified_log_axes(ax, ylim)
    ax.set_xlabel('Transition Energy $\\hbar\\omega_{ij} / h$ (GHz)')
    ax.set_ylabel('Transition Rate $\\Gamma_{j \\to i}$ (MHz)')
    ax.set_title(f'Environment Overlay: Transmission Line vs BPF (T = {temp_str})')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved overlay plot: {filename}")


def main():
    print("=" * 70)
    print("FLUXONIUM TRANSITION RATE SIMULATION (qutip & scqubits)")
    print("=" * 70)

    # 1. Setup system
    system_data = setup_fluxonium_system()

    # 2. Environmental circuit parameters
    EC_J = system_data['EC'] * 1e9 * const.h
    C_sigma = (const.e**2) / (2.0 * EC_J)

    env_params = {
        'C_sigma': C_sigma,       
        'Q_cap': 1e6,             
        'Cc': 5e-15,             
        'Z0': 50.0,  
        # --- 更新為 N=6, 0dB, 5.9GHz, 500MHz 的 Multi-stage BPF 參數 ---
        'f_filter_hz': 5.9e9,     # 濾波器中心頻率 5.9 GHz
        'bw_hz': 500e6,           # 濾波器頻寬 500 MHz 
        'N_order': 6,             # N=6 階濾波器 
        
        # 根據文獻 Table I: 0 dB, N=6 的 g 係數乘積 (g0g1, g1g2, g2g3, g3g4, g4g5, g5g6, g6g7)
        'g_prods': [0.518, 0.732, 2.732, 3.732, 2.732, 0.732, 0.518], 
        
        # 0 dB 為對稱型濾波器，讀取諧振腔必須耦合在中間級 (Middle stage) 以獲得最佳保護
        'coupling_stage': 3       # 設為第 3 級 (或第 4 級亦可)
    }

    print("\nEnvironmental & Coupling Parameters:")
    print(f"  C_sigma (total qubit cap) = {C_sigma*1e15:.2f} fF")
    print(f"  Q_cap (dielectric Q)     = {env_params['Q_cap']:.1e}")
    print(f"  Cc (coupling cap)        = {env_params['Cc']*1e15:.2f} fF")
    print(f"  Z0 (line impedance)      = {env_params['Z0']} Ohm")
    print(f"  BPF center frequency     = {env_params['f_filter_hz']/1e9} GHz")

    # Output directory for plots
    out_dir = "transition_rate_plots"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Calculate transition rates for all combinations
    environments = [
        ('uncoupled', 'Uncoupled (Intrinsic)'),
        ('transmission_line', 'Transmission Line'),
        ('multi_stage_bpf', 'N=5 Multi-Stage BPF')
    ]
    temperatures = [
        (0.0, '0K'),
        (0.070, '70mK')
    ]

    all_results = {}
    all_rates_list = []

    print("\n" + "=" * 70)
    print("SIMULATION RESULTS TABLE")
    print("=" * 70)

    for env_key, env_name in environments:
        all_results[env_key] = {}
        for T_k, T_str in temperatures:
            rates = calculate_transition_rates(system_data, T_k, env_key, env_params)
            all_results[env_key][T_str] = rates

            for t in rates:
                if t['rate_mhz'] > 0:
                    all_rates_list.append(t['rate_mhz'])

            print(f"\n---> Environment: {env_name} | Temperature: {T_str}")
            print(f"{'Transition':<12} {'Type':<12} {'Energy (GHz)':<15} {'Rate (MHz)':<15}")
            print("-" * 55)
            for t in sorted(rates, key=lambda x: x['label']):
                print(f"{t['label']:<12} {t['transition_type']:<12} {t['energy_ghz']:<15.4f} {t['rate_mhz']:<15.4e}")

    # Determine global Y-axis bounds rounded to power-of-10 decade boundaries
    if all_rates_list:
        min_rate = min(all_rates_list)
        max_rate = max(all_rates_list)
        y_min_pow = int(np.floor(np.log10(min_rate)))
        y_max_pow = int(np.ceil(np.log10(max_rate)))
        global_ylim = (10.0**y_min_pow, 10.0**y_max_pow)
    else:
        global_ylim = (1e-41, 1e4)

    print("\n" + "=" * 70)
    print(f"GLOBAL UNIFIED Y-AXIS SCALE: {global_ylim[0]:.1e} to {global_ylim[1]:.1e} MHz")
    print("=" * 70)

    # 4. Generate Base Plots with unified Y-limits
    for env_key, env_name in environments:
        for _, T_str in temperatures:
            rates = all_results[env_key][T_str]
            filename = os.path.join(out_dir, f"1_base_{env_key}_{T_str}.png")
            create_base_plot(rates, env_name, T_str, filename, ylim=global_ylim)

    # 5. Generate Overlay Plots with unified Y-limits
    print("\n" + "=" * 70)
    print("GENERATING OVERLAY PLOTS")
    print("=" * 70)

    # 0K vs 70mK Overlays for each environment (3 plots)
    for env_key, env_name in environments:
        filename = os.path.join(out_dir, f"2_overlay_temp_{env_key}_0K_vs_70mK.png")
        create_temp_overlay_plot(all_results[env_key]['0K'], all_results[env_key]['70mK'], env_name, filename, ylim=global_ylim)

    # TL vs BPF Overlays for each temperature (2 plots)
    for _, T_str in temperatures:
        filename = os.path.join(out_dir, f"3_overlay_env_TL_vs_BPF_{T_str}.png")
        create_env_overlay_plot(all_results['transmission_line'][T_str],
                                all_results['multi_stage_bpf'][T_str],
                                T_str, filename, ylim=global_ylim)


if __name__ == '__main__':
    main()
