"""Run 0-3-2 STIRAP using the functions in fluxonium_rates.py.

Keep the original user code in the same directory and save it as
`fluxonium_rates.py`. This script only calls the existing functions
`setup_fluxonium_system()` and `calculate_transition_rates()`.
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import fluxonium_bpf as model

def check_stirap_conditions(
    tlist,
    Omega_p,
    Omega_s,
    t_pump,
    t_stokes,
    omega_p=None,
    omega_s=None,
    E=None,
    n03=None,
    n32=None,
    adiabatic_threshold=10.0,
    resonance_tolerance=1e-6,
):
    """
    Check the basic STIRAP and adiabatic conditions.

    For 0-3-2 STIRAP:
        Pump:   |0> <-> |3>
        Stokes: |2> <-> |3>
        Target: |0> -> |2>

    All Rabi frequencies must be angular frequencies (rad/ns).
    Energies E must be in GHz.
    """

    report = {}

    t = np.asarray(tlist, dtype=float)
    Op = np.asarray(Omega_p, dtype=float)
    Os = np.asarray(Omega_s, dtype=float)

    # --------------------------------------------------
    # 1. Basic input validation
    # --------------------------------------------------
    if (
        t.ndim != 1
        or Op.shape != t.shape
        or Os.shape != t.shape
        or len(t) < 3
    ):
        raise ValueError("Invalid time or Rabi-frequency arrays.")

    if not (
        np.all(np.isfinite(t))
        and np.all(np.isfinite(Op))
        and np.all(np.isfinite(Os))
    ):
        raise ValueError("Inputs contain NaN or infinite values.")

    if np.any(np.diff(t) <= 0):
        raise ValueError("tlist must be strictly increasing.")

    if np.any(Op < 0) or np.any(Os < 0):
        raise ValueError("Rabi envelopes must be non-negative.")

    if np.max(Op) <= 0 or np.max(Os) <= 0:
        raise ValueError("Both Pump and Stokes pulses must be nonzero.")

    # --------------------------------------------------
    # 2. Counter-intuitive pulse sequence
    # --------------------------------------------------
    pulse_order = t_stokes < t_pump

    report["pulse_order"] = pulse_order

    # --------------------------------------------------
    # 3. Pulse overlap
    # --------------------------------------------------
    overlap = np.trapezoid(np.minimum(Op, Os), t)
    smaller_area = min(
        np.trapezoid(Op, t),
        np.trapezoid(Os, t)
    )

    overlap_ratio = overlap / smaller_area if smaller_area > 0 else 0.0

    report["overlap_ratio"] = overlap_ratio
    report["pulse_overlap"] = overlap_ratio > 0.01

    # --------------------------------------------------
    # 4. Nonzero transition matrix elements
    # --------------------------------------------------
    if n03 is not None and n32 is not None:
        report["valid_matrix_elements"] = (
            abs(n03) > 1e-14 and abs(n32) > 1e-14
        )

    # --------------------------------------------------
    # 5. Two-photon resonance
    # --------------------------------------------------
    if omega_p is not None and omega_s is not None and E is not None:

        E = np.asarray(E, dtype=float)

        if len(E) < 4:
            raise ValueError("At least four energy levels are required.")

        # Angular frequency units: rad/ns
        omega_02 = 2.0 * np.pi * (E[2] - E[0])

        delta_2ph = omega_p - omega_s - omega_02

        report["two_photon_detuning_rad_ns"] = delta_2ph

        report["two_photon_resonance"] = (
            abs(delta_2ph) < resonance_tolerance
        )

    # --------------------------------------------------
    # 6. Dark-state mixing angle
    # --------------------------------------------------
    Omega_rms = np.sqrt(Op**2 + Os**2)

    theta = np.arctan2(Op, Os)

    theta_dot = np.gradient(theta, t)

    # --------------------------------------------------
    # 7. Local adiabatic condition
    #
    # Omega_rms >> |d(theta)/dt|
    # --------------------------------------------------
    active = Omega_rms > np.max(Omega_rms) * 1e-6

    local_ratio = (
        Omega_rms[active] /
        np.maximum(np.abs(theta_dot[active]), 1e-30)
    )

    min_local_ratio = np.min(local_ratio)

    local_ok = min_local_ratio >= adiabatic_threshold

    report["min_local_adiabatic_ratio"] = min_local_ratio
    report["local_adiabatic"] = local_ok

    # --------------------------------------------------
    # 8. Global adiabatic condition
    #
    # Integral(Omega_rms dt) >> pi/2
    # --------------------------------------------------
    A = np.trapezoid(Omega_rms, t)

    global_ratio = A / (np.pi / 2.0)

    global_ok = global_ratio >= adiabatic_threshold

    report["adiabatic_integral"] = A
    report["global_adiabatic_ratio"] = global_ratio
    report["global_adiabatic"] = global_ok

    # --------------------------------------------------
    # 9. Initial and final dark-state mixing angles
    # --------------------------------------------------
    theta_initial = theta[0]
    theta_final = theta[-1]

    endpoint_ok = (
        theta_initial < 0.1
        and theta_final > np.pi / 2.0 - 0.1
    )

    report["theta_initial"] = theta_initial
    report["theta_final"] = theta_final
    report["dark_state_transfer"] = endpoint_ok

    # --------------------------------------------------
    # 10. Overall evaluation
    # --------------------------------------------------
    required_checks = [
        report["pulse_order"],
        report["pulse_overlap"],
        report["local_adiabatic"],
        report["global_adiabatic"],
        report["dark_state_transfer"],
    ]

    if "valid_matrix_elements" in report:
        required_checks.append(report["valid_matrix_elements"])

    if "two_photon_resonance" in report:
        required_checks.append(report["two_photon_resonance"])

    report["STIRAP_satisfied"] = all(required_checks)

    # --------------------------------------------------
    # 11. Print report
    # --------------------------------------------------
    print("\n" + "=" * 55)
    print("       STIRAP CONDITION CHECK (0-3-2)")
    print("=" * 55)

    for key, value in report.items():

        if isinstance(value, (bool, np.bool_)):
            status = "PASS" if value else "FAIL"
            print(f"{key:35s}: {status}")

        elif isinstance(value, (float, np.floating)):
            print(f"{key:35s}: {value:.6g}")

        else:
            print(f"{key:35s}: {value}")

    print("=" * 55)

    if report["STIRAP_satisfied"]:
        print("RESULT: STIRAP CONDITIONS SATISFIED")
    else:
        print("RESULT: STIRAP CONDITIONS NOT SATISFIED")

    print("=" * 55 + "\n")

    return report

# ----------------------------------------------------------------------
# 1. Fluxonium eigen-system: call the existing user function
# ----------------------------------------------------------------------
system_data = model.setup_fluxonium_system()
N = system_data["N_LEVELS"]
if N < 4:
    raise ValueError("0-3-2 STIRAP requires at least 4 energy levels.")

# ----------------------------------------------------------------------
# 2. Transmission-line transition rates: call the existing user function
# ----------------------------------------------------------------------
EC_J = system_data["EC"] * 1e9 * model.const.h
C_sigma = model.const.e**2 / (2.0 * EC_J)

env_params = {
    'C_sigma': C_sigma,       
    'Q_cap': 1e6,             
    'Cc': 1.1e-15,             
    'Z0': 50.0,  
    # --- 更新為 N=6, 0dB, 5.9GHz, 500MHz 的 Multi-stage BPF 參數 ---
    'f_filter_hz': 5.8e9,     # 濾波器中心頻率 5.9 GHz
    'bw_hz': 1000e6,           # 濾波器頻寬 500 MHz 
    'N_order': 6,             # N=6 階濾波器 
    
    # 根據文獻 Table I: 0 dB, N=6 的 g 係數乘積 (g0g1, g1g2, g2g3, g3g4, g4g5, g5g6, g6g7)
    'g_prods': [0.518, 0.732, 2.732, 3.732, 2.732, 0.732, 0.518], 
    #'g_prods': [0.765, 1.414,3.414,1.414,0.765], 
    #'g_prods': [1.0,2.0,2.0,1.0], 
    #'g_prods': [1.414,2.0,1.414], 
    
    # 0 dB 為對稱型濾波器，讀取諧振腔必須耦合在中間級 (Middle stage) 以獲得最佳保護
    'coupling_stage': 3       # 設為第 3 級 (或第 4 級亦可)
}
#env_type = "multi_stage_bpf"
env_type = "transmission_line"
T_kelvin = 0.030
rates = model.calculate_transition_rates(
    system_data=system_data,
    T_kelvin=T_kelvin,
    env_type=env_type,
    params=env_params,
)

print("\nTransmission-line transition rates at", T_kelvin * 1e3, "mK")
for item in rates:
    print(
        f"{item['from']} -> {item['to']}: "
        f"{item['rate_mhz']:.6g} MHz"
    )


# ----------------------------------------------------------------------
# 3. Convert the transition rates into QuTiP collapse operators
# ----------------------------------------------------------------------
# Rates returned by calculate_transition_rates() are in MHz.
# Since t is in ns: 1 MHz = 1e-3 ns^-1. Do not multiply by 2*pi.
c_ops = []
for item in rates:
    initial = int(item["from"])
    final = int(item["to"])
    gamma_ns = float(item["rate_mhz"]) * 1e-3

    if gamma_ns > 0.0:
        jump_op = qt.basis(N, final) * qt.basis(N, initial).dag()
        c_ops.append(np.sqrt(gamma_ns) * jump_op)

print(f"\nNumber of collapse operators = {len(c_ops)}")


# ----------------------------------------------------------------------
# 4. 0-3-2 STIRAP Hamiltonian and requested pulse parameters
# ----------------------------------------------------------------------
# Energies from scqubits are in GHz. With time in ns, multiply by 2*pi
# to obtain angular frequencies in rad/ns.
H0 = 2.0 * np.pi * system_data["H_qutip"]
n_op = system_data["n_qutip"]
n_array = n_op.full()
E = np.asarray(system_data["evals_ghz"], dtype=float)

# Resonant carrier angular frequencies (rad/ns).
omega_p = 2.0 * np.pi * (E[3] - E[0])  # Pump: 0 <-> 3
omega_s = 2.0 * np.pi * (E[3] - E[2])  # Stokes: 2 <-> 3

# Parameters read from the reference figure.
# The pulse centers are symmetric about t_mid, with Stokes arriving first.
OMEGA0_OVER_2PI_MHZ = 78
SIGMA_NS = 35
TAU_NS = 42
T_MID_NS = 114

# Use the Gaussian convention that matches the reference plot:
# Omega(t) = Omega0 * exp(-((t - t_center)/sigma)^2).
t_stokes = T_MID_NS - TAU_NS / 2.0
t_pump = T_MID_NS + TAU_NS / 2.0

# Requested peak Rabi angular frequency: 20 MHz = 0.020 GHz.
omega0_rad_ns = 2.0 * np.pi * (OMEGA0_OVER_2PI_MHZ * 1e-3)

# For H_drive = A(t) * n * cos(omega_d*t), the resonant Rabi angular
# frequency on transition |i> <-> |j> is A(t)*|<j|n|i>| under RWA.
# Normalize the two drive amplitudes independently so BOTH transitions
# have the requested peak Omega0/(2*pi) = 20 MHz.
n03 = abs(n_array[3, 0])
n32 = abs(n_array[3, 2])
print(n03,n32)
if n03 < 1e-14 or n32 < 1e-14:
    raise ValueError(
        "The n-operator matrix element for transition 0<->3 or 2<->3 "
        "is approximately zero; cannot normalize the requested Rabi rate."
    )

A_pump = omega0_rad_ns / n03
A_stokes = omega0_rad_ns / n32


def gaussian_envelope(t, center, sigma):
    return np.exp(-((t - center) / sigma) ** 2)


def drive_coefficient(t, args):
    pump_env = gaussian_envelope(t, args["t_pump"], args["sigma"])
    stokes_env = gaussian_envelope(t, args["t_stokes"], args["sigma"])

    pump_drive = args["A_pump"] * pump_env * np.cos(args["omega_p"] * t)
    stokes_drive = args["A_stokes"] * stokes_env * np.cos(args["omega_s"] * t)
    return pump_drive + stokes_drive


H_tot = [H0, [n_op, drive_coefficient]]
args = {
    "A_pump": A_pump,
    "A_stokes": A_stokes,
    "omega_p": omega_p,
    "omega_s": omega_s,
    "t_pump": t_pump,
    "t_stokes": t_stokes,
    "sigma": SIGMA_NS,
}


# ----------------------------------------------------------------------
# 5. Initial state, observables, and Lindblad master-equation solver
# ----------------------------------------------------------------------
psi0 = qt.basis(N, 0)

P0 = qt.basis(N, 0).proj()  # Initial state |0>
P2 = qt.basis(N, 2).proj()  # Target state |2>
P3 = qt.basis(N, 3).proj()  # Intermediate state |3>
P_other = qt.qeye(N) - P0 - P2 - P3

tlist = np.linspace(0.0, T_MID_NS*2, 6001)  # ns

# Resolve the GHz carrier oscillations in the lab-frame Hamiltonian.
max_omega = max(abs(omega_p), abs(omega_s))
carrier_period = 2.0 * np.pi / max_omega
max_step = min(tlist[1] - tlist[0], carrier_period / 1000.0)

options = {
    "nsteps": 100000,
    "max_step": max_step,
    "rtol": 1e-7,
    "atol": 1e-9,
}


# --------------------------------------------------
# STIRAP parameter verification
# --------------------------------------------------

Omega_s_check = omega0_rad_ns * gaussian_envelope(
    tlist, t_stokes, SIGMA_NS
)

Omega_p_check = omega0_rad_ns * gaussian_envelope(
    tlist, t_pump, SIGMA_NS
)

stirap_report = check_stirap_conditions(
    tlist=tlist,
    Omega_p=Omega_p_check,
    Omega_s=Omega_s_check,

    t_pump=t_pump,
    t_stokes=t_stokes,

    omega_p=omega_p,
    omega_s=omega_s,
    E=E,

    n03=n03,
    n32=n32,

    adiabatic_threshold=10.0
)
"""
if not stirap_report["STIRAP_satisfied"]:
    raise RuntimeError(
        "STIRAP conditions failed. "
        "Please adjust the pulse parameters."
    )
"""
result = qt.mesolve(
    H_tot,
    psi0,
    tlist,
    c_ops,
    e_ops=[P0, P3, P2, P_other],
    args=args,
    options=options,
)

P0_t, P3_t, P2_t, Pother_t = [np.real(np.asarray(x)) for x in result.expect]

# For transfer to the target basis state |2>, F = <2|rho(t_final)|2> = P2(t_final).
fidelity = float(P2_t[-1])

print("\nSTIRAP result (0 -> 3 -> 2)")
print(f"Final P0       = {P0_t[-1]:.6f}")
print(f"Final P3       = {P3_t[-1]:.6f}")
print(f"Final P2       = {P2_t[-1]:.6f}  (fidelity)")
print(f"Final P_other  = {Pother_t[-1]:.6f}")
print(f"Maximum P3     = {np.max(P3_t):.6f}")
print(f"Peak Rabi rate = {OMEGA0_OVER_2PI_MHZ:.3f} MHz")
print(f"Stokes center  = {t_stokes:.2f} ns")
print(f"Pump center    = {t_pump:.2f} ns")


# ----------------------------------------------------------------------
# 6. Calculate the displayed Rabi-frequency envelopes (MHz)
# ----------------------------------------------------------------------
Omega_s_mhz = OMEGA0_OVER_2PI_MHZ * gaussian_envelope(
    tlist, t_stokes, SIGMA_NS
)
Omega_p_mhz = OMEGA0_OVER_2PI_MHZ * gaussian_envelope(
    tlist, t_pump, SIGMA_NS
)


# ----------------------------------------------------------------------
# 7. Save the two requested plots separately
# ----------------------------------------------------------------------
# Plot 1: Stokes and Pump Rabi-frequency envelopes.
fig1, ax1 = plt.subplots(figsize=(7.2, 2.8))
ax1.plot(tlist, Omega_s_mhz, "--", label=r"$\Omega_s$")
ax1.plot(tlist, Omega_p_mhz, "--", label=r"$\Omega_p$")
ax1.set_ylabel("Rabi frequency [MHz]")
ax1.set_xlabel("Time [ns]")
ax1.set_xlim(0.0, T_MID_NS*2)
ax1.set_ylim(0.0, OMEGA0_OVER_2PI_MHZ+2)
ax1.legend(loc="upper right")
ax1.grid(False)
fig1.tight_layout()
fig1.savefig(f"stirap_rabi_pulses_{env_type}.png", dpi=300, bbox_inches="tight")


# Plot 2: Population transfer. The p1/p2/p3 labels follow the reference:
# p1 = |0> (initial), p2 = |3> (intermediate), p3 = |2> (target).
fig2, ax2 = plt.subplots(figsize=(7.2, 2.8))
ax2.plot(tlist, P0_t, label=r"$p_{|0\rangle}$")
ax2.plot(tlist, P3_t, label=r"$p_{|3\rangle}$")
ax2.plot(tlist, P2_t, label=r"$p_{|2\rangle}$")
ax2.set_xlabel("Time [ns]")
ax2.set_ylabel("Population")
ax2.set_xlim(0.0, T_MID_NS*2)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc="center right")
ax2.grid(False)
fig2.tight_layout()
fig2.savefig(f"stirap_populations_{env_type}.png", dpi=300, bbox_inches="tight")


print("\nSaved figures:")
print("  stirap_rabi_pulses.png")
print("  stirap_populations.png")

#plt.show()
