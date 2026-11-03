"""Flat-top Gaussian pulse + weighted Legendre QR basis + fidelity optimization.

Units
-----
Time: ns
Omega: rad/ns
Omega0 input: MHz in Omega0/(2*pi)

The fidelity callback must have the signature
    fidelity_fn(Omega_array, theta_array, T_ns) -> scalar
where Omega_array and theta_array are sampled on the same time grid returned
by this module.

The default ramp calibration below is only an illustrative placeholder.
Replace it with the measured/calibrated instrument response before using
results as physical predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any
import os
import sys
import multiprocessing as mp
import concurrent.futures

# Set thread environment variables before linear algebra libraries load
# to ensure maximum parallel efficiency across CPU cores in multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import qutip as qt

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fluxonium_bpf as model


# ---------------------------------------------------------------------------
# User-provided pulse settings
# ---------------------------------------------------------------------------
T_NS = 200.0
OMEGA0_OVER_2PI_MHZ = 20.0
OMEGA0_RAD_NS = 2.0 * np.pi * (OMEGA0_OVER_2PI_MHZ * 1e-3)




# ---------------------------------------------------------------------------
# Physical System Setup and Precomputation Cache
# ---------------------------------------------------------------------------
_SYSTEM_CACHE: Optional[dict[str, Any]] = None


def get_stirap_system_cache(
    env_type: str = "transmission_line",
    T_kelvin: float = 0.030,
) -> dict[str, Any]:
    """Initialize and cache the Fluxonium system, transition rates, and collapse operators.

    This ensures system diagonalization and Fermi golden rule rate calculations
    are performed only once across all optimization iterations and worker processes.
    """
    global _SYSTEM_CACHE
    if _SYSTEM_CACHE is not None and _SYSTEM_CACHE.get("env_type") == env_type and _SYSTEM_CACHE.get("T_kelvin") == T_kelvin:
        return _SYSTEM_CACHE

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
        raise ValueError(
            "Matrix elements n03 or n32 are approximately zero; cannot normalize Rabi rates."
        )

    _SYSTEM_CACHE = {
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
    return _SYSTEM_CACHE


def simulate_stirap_arbitrary_pulse(
    t_grid: np.ndarray,
    Omega_p: np.ndarray,
    Omega_s: np.ndarray,
    *,
    system_cache: Optional[dict[str, Any]] = None,
    solver_options: Optional[dict[str, Any]] = None,
    e_ops: Optional[list[Any]] = None,
) -> qt.Result:
    """Simulate 0-3-2 STIRAP dynamics under arbitrary pump and Stokes envelopes.

    Omega_p and Omega_s are Rabi frequencies in rad/ns evaluated at t_grid (ns).
    Continuous driving amplitudes are constructed via CubicSpline interpolation.
    """
    if system_cache is None:
        system_cache = get_stirap_system_cache()

    N = system_cache["N"]
    H0 = system_cache["H0"]
    n_op = system_cache["n_op"]
    omega_p = system_cache["omega_p"]
    omega_s = system_cache["omega_s"]
    n03 = system_cache["n03"]
    n32 = system_cache["n32"]
    c_ops = system_cache["c_ops"]

    t_grid = np.asarray(t_grid, dtype=float)
    Omega_p = np.asarray(Omega_p, dtype=float)
    Omega_s = np.asarray(Omega_s, dtype=float)

    spline_p = CubicSpline(t_grid, Omega_p)
    spline_s = CubicSpline(t_grid, Omega_s)

    def drive_coefficient(t: float, *unused_args: Any, **unused_kwargs: Any) -> float:
        # A_pump(t) = Omega_p(t) / n03; A_stokes(t) = Omega_s(t) / n32
        p_val = float(spline_p(t)) / n03
        s_val = float(spline_s(t)) / n32
        return p_val * np.cos(omega_p * t) + s_val * np.cos(omega_s * t)

    H_tot = [H0, [n_op, drive_coefficient]]
    psi0 = qt.basis(N, 0)

    if e_ops is None:
        P0 = qt.basis(N, 0).proj()
        P2 = qt.basis(N, 2).proj()
        P3 = qt.basis(N, 3).proj()
        P_other = qt.qeye(N) - P0 - P2 - P3
        e_ops = [P0, P3, P2, P_other]

    max_omega = max(abs(omega_p), abs(omega_s))
    carrier_period = 2.0 * np.pi / max_omega
    # Default solver options matching run_stirap.py
    default_max_step = min(t_grid[1] - t_grid[0], carrier_period / 1000.0)

    options = {
        "nsteps": 100000,
        "max_step": default_max_step,
        "rtol": 1e-7,
        "atol": 1e-9,
    }
    if solver_options is not None:
        options.update(solver_options)

    result = qt.mesolve(
        H_tot,
        psi0,
        t_grid,
        c_ops,
        e_ops=e_ops,
        options=options,
    )
    return result


def calculate_stirap_fidelity(
    Omega_array: np.ndarray,
    theta_array: np.ndarray,
    T_ns: float,
    *,
    t_grid: Optional[np.ndarray] = None,
    system_cache: Optional[dict[str, Any]] = None,
    solver_options: Optional[dict[str, Any]] = None,
) -> float:
    """Calculate the STIRAP fidelity (target state |2> population at t=T).

    Omega_p(t) = Omega(t) * sin(theta(t))
    Omega_s(t) = Omega(t) * cos(theta(t))
    """
    if t_grid is None:
        t_grid = np.linspace(0.0, T_ns, len(Omega_array))

    Omega_p = Omega_array * np.sin(theta_array)
    Omega_s = Omega_array * np.cos(theta_array)

    if system_cache is None:
        system_cache = get_stirap_system_cache()

    N = system_cache["N"]
    P2 = qt.basis(N, 2).proj()

    result = simulate_stirap_arbitrary_pulse(
        t_grid=t_grid,
        Omega_p=Omega_p,
        Omega_s=Omega_s,
        system_cache=system_cache,
        solver_options=solver_options,
        e_ops=[P2],
    )
    p2_t = np.real(np.asarray(result.expect[0]))
    return float(p2_t[-1])


# ---------------------------------------------------------------------------
# Instrument ramp calibration
# ---------------------------------------------------------------------------
def power_law_ramp_calibration(
    omega0_over_2pi_mhz: float,
    *,
    reference_omega_mhz: float = 20.0,
    reference_rise_time_ns: float = 10.0,
    reference_sigma_ns: float = 2.5,
    rise_time_exponent: float = 0.25,
    sigma_exponent: float = 0.25,
) -> tuple[float, float]:
    """Return (Gaussian ramp duration, Gaussian sigma), both in ns.

    This example makes both quantities amplitude-dependent:
        t_r = t_r_ref * (Omega0/Omega_ref)**rise_time_exponent
        sigma = sigma_ref * (Omega0/Omega_ref)**sigma_exponent

    The numerical reference values and exponents are placeholders, not an
    assumed physical law for a particular instrument. Replace this function
    with a calibration curve or interpolation of measured data as needed.
    """
    if omega0_over_2pi_mhz <= 0.0 or reference_omega_mhz <= 0.0:
        raise ValueError("Omega0 and reference_omega_mhz must be positive.")

    ratio = omega0_over_2pi_mhz / reference_omega_mhz
    rise_time_ns = reference_rise_time_ns * ratio**rise_time_exponent
    sigma_ns = reference_sigma_ns * ratio**sigma_exponent
    return float(rise_time_ns), float(sigma_ns)


def flat_top_gaussian_envelope(
    t_ns: np.ndarray,
    T_ns: float,
    omega0_rad_ns: float,
    rise_time_ns: float,
    sigma_ns: float,
) -> np.ndarray:
    """Flat top with Gaussian leading/trailing ramps.

    The left Gaussian is centered at t=rise_time_ns and the right Gaussian
    at t=T-rise_time_ns. Thus each ramp joins the flat top smoothly with
    value Omega0 and zero slope. The Gaussian tails are truncated at t=0,T;
    for rise_time_ns=4*sigma_ns the endpoint amplitude is exp(-8)*Omega0.

    Here rise_time_ns means the ramp *duration* from the pulse boundary to
    the flat section, not the conventional 10%-90% rise time.
    """
    t_ns = np.asarray(t_ns, dtype=float)
    if T_ns <= 0.0:
        raise ValueError("T_ns must be positive.")
    if omega0_rad_ns <= 0.0:
        raise ValueError("omega0_rad_ns must be positive.")
    if rise_time_ns <= 0.0 or sigma_ns <= 0.0:
        raise ValueError("rise_time_ns and sigma_ns must be positive.")
    if 2.0 * rise_time_ns >= T_ns:
        raise ValueError("The two ramps overlap: require 2*rise_time_ns < T_ns.")
    if np.any(t_ns < 0.0) or np.any(t_ns > T_ns):
        raise ValueError("All time samples must lie in [0, T_ns].")

    omega = np.full_like(t_ns, omega0_rad_ns, dtype=float)
    left = t_ns < rise_time_ns
    right = t_ns > (T_ns - rise_time_ns)

    omega[left] = omega0_rad_ns * np.exp(
        -0.5 * ((t_ns[left] - rise_time_ns) / sigma_ns) ** 2
    )
    omega[right] = omega0_rad_ns * np.exp(
        -0.5 * ((t_ns[right] - (T_ns - rise_time_ns)) / sigma_ns) ** 2
    )
    return omega


# ---------------------------------------------------------------------------
# Omega Iteration from Theta
# ---------------------------------------------------------------------------
def update_omega_from_theta(
    Omega: np.ndarray,
    theta: np.ndarray,
    n03: float,
    n32: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Update Omega envelope using theta and transition dipole matrix elements.

    Correction formula:
        Omega_new = Omega * min( (n03 / n32) / sin(theta), 1 / cos(theta) )

    Numerical protection with `eps` prevents division by zero at endpoints
    (e.g., sin(theta)=0 at theta=0 or cos(theta)=0 at theta=pi/2).
    """
    Omega = np.asarray(Omega, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if Omega.shape != theta.shape:
        raise ValueError(
            f"Omega and theta must have the same shape, got {Omega.shape} and {theta.shape}."
        )
    if n03 <= 0.0 or n32 <= 0.0:
        raise ValueError("n03 and n32 must be strictly positive.")

    sin_theta = np.maximum(np.sin(theta), eps)
    cos_theta = np.maximum(np.cos(theta), eps)
    term1 = (n03 / n32) / sin_theta
    term2 = 1.0 / cos_theta
    factor = np.minimum(term1, term2)
    return Omega * factor


# ---------------------------------------------------------------------------
# Numerical integration utilities (composite trapezoidal rule)
# ---------------------------------------------------------------------------
def trapezoid_weights(t: np.ndarray) -> np.ndarray:
    """Quadrature weights whose dot product equals np.trapezoid(y, t)."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1-D array with at least two points.")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("t must be strictly increasing.")

    weights = np.empty_like(t)
    weights[0] = 0.5 * dt[0]
    weights[-1] = 0.5 * dt[-1]
    if t.size > 2:
        weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return weights


def cumulative_trapezoid(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with the same grid convention."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.shape != t.shape:
        raise ValueError("y and t must have the same shape.")
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(t))
    return out


# ---------------------------------------------------------------------------
# Legendre seeds + weighted QR decomposition
# ---------------------------------------------------------------------------
def build_weighted_legendre_basis(
    t_ns: np.ndarray,
    Omega: np.ndarray,
    max_degree: int,
) -> dict[str, Any]:
    """Build phi_0,...,phi_N by weighted QR of Legendre seed functions.

    The discrete inner product is
        <f,g>_Omega = sum_i q_i f(t_i) g(t_i) Omega(t_i),
    where q_i are composite-trapezoid quadrature weights.

    Since the first seed is P_0=1, the first QR basis function is
        phi_0(t_i) = 1/sqrt(W),  W = sum_i q_i Omega(t_i),
    and all higher basis functions have zero weighted integral (up to
    floating-point roundoff).
    """
    t_ns = np.asarray(t_ns, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    if t_ns.ndim != 1 or Omega.shape != t_ns.shape:
        raise ValueError("t_ns and Omega must be 1-D arrays of the same shape.")
    if not isinstance(max_degree, (int, np.integer)) or max_degree < 0:
        raise ValueError("max_degree must be a non-negative integer.")
    if max_degree + 1 >= t_ns.size:
        raise ValueError("Need more time samples than basis functions.")
    if np.any(~np.isfinite(Omega)) or np.any(Omega <= 0.0):
        raise ValueError("Omega must be finite and strictly positive on the grid.")

    q = trapezoid_weights(t_ns)
    weighted_measure = q * Omega
    W = float(np.sum(weighted_measure))
    if not np.isfinite(W) or W <= 0.0:
        raise ValueError("Weighted pulse area W must be positive and finite.")

    # Map [0,T] to [-1,1] and evaluate P_0,...,P_N.
    x = 2.0 * (t_ns - t_ns[0]) / (t_ns[-1] - t_ns[0]) - 1.0
    P = legvander(x, max_degree)  # shape: (n_time, max_degree+1)

    sqrt_measure = np.sqrt(weighted_measure)
    A = sqrt_measure[:, None] * P
    Q, R = np.linalg.qr(A, mode="reduced")

    # Make QR signs reproducible: choose non-negative diagonal entries of R.
    diag_r = np.diag(R)
    scale = max(float(np.max(np.abs(diag_r))), 1.0)
    rank_tol = np.finfo(float).eps * max(A.shape) * scale
    if np.any(np.abs(diag_r) <= rank_tol):
        raise np.linalg.LinAlgError(
            "Weighted Legendre matrix is numerically rank deficient. "
            "Reduce max_degree or increase the number/quality of grid points."
        )
    for j, rjj in enumerate(diag_r):
        if rjj < 0.0:
            Q[:, j] *= -1.0
            R[j, :] *= -1.0

    Phi = Q / sqrt_measure[:, None]
    gram = Phi.T @ (weighted_measure[:, None] * Phi)
    orthogonality_error = float(np.max(np.abs(gram - np.eye(max_degree + 1))))

    return {
        "Phi": Phi,
        "seed_matrix": P,
        "R": R,
        "quadrature_weights": q,
        "weighted_measure": weighted_measure,
        "W": W,
        "orthogonality_error": orthogonality_error,
        "gram_matrix": gram,
    }


def reconstruct_eta_theta(
    t_ns: np.ndarray,
    Omega: np.ndarray,
    basis: dict[str, Any],
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct eta and theta for coefficients a_1,...,a_N.

    eta(t) = 1/W + sum_{n=1}^N a_n phi_n(t)
    theta(t) = (pi/2) * integral_0^t eta(s) Omega(s) ds

    The hard endpoint constraint is built into the parameterization.
    """
    t_ns = np.asarray(t_ns, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    Phi = basis["Phi"]
    N = Phi.shape[1] - 1
    if coefficients.shape != (N,):
        raise ValueError(f"Expected {N} coefficients, got shape {coefficients.shape}.")

    W = float(basis["W"])
    eta = np.full_like(t_ns, 1.0 / W, dtype=float)
    if N:
        eta += Phi[:, 1:] @ coefficients

    theta = (np.pi / 2.0) * cumulative_trapezoid(eta * Omega, t_ns)
    return eta, theta


def project_eta_to_basis(
    eta_prev: np.ndarray,
    basis_new: dict[str, Any],
) -> np.ndarray:
    """Project an existing eta(t) envelope onto a new weighted Legendre basis.

    This preserves the pulse shape and high fidelity across outer iteration rounds
    when Omega(t) is updated and the basis is reconstructed:
        a_n = <phi_n^{new}, eta_{prev}>_{Omega_{new}}
    """
    Phi = basis_new["Phi"]
    weighted_measure = basis_new["weighted_measure"]
    # Phi[:, 1:] has shape (n_time, N_free)
    # Project eta_prev onto phi_1,...,phi_N using inner product with weighted_measure
    coeffs = Phi[:, 1:].T @ (weighted_measure * np.asarray(eta_prev, dtype=float))
    return coeffs


# ---------------------------------------------------------------------------
# Optimizers: objective is maximized. Add alternative optimizers with the
# same call signature to make replacing gradient ascent straightforward.
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    x: np.ndarray
    fun: float
    nit: int
    success: bool
    message: str
    history: list[dict[str, float]]


_PARALLEL_OBJECTIVE: Optional[Callable[[np.ndarray], float]] = None


def _parallel_worker_init(objective: Callable[[np.ndarray], float]) -> None:
    global _PARALLEL_OBJECTIVE
    _PARALLEL_OBJECTIVE = objective


def _parallel_worker_evaluate(point: np.ndarray) -> float:
    global _PARALLEL_OBJECTIVE
    if _PARALLEL_OBJECTIVE is None:
        raise RuntimeError("Global objective was not initialized in worker.")
    return float(_PARALLEL_OBJECTIVE(point))


def parallel_evaluate_points(
    objective: Callable[[np.ndarray], float],
    points: list[np.ndarray],
    max_workers: Optional[int] = None,
) -> list[float]:
    """Cross-platform parallel evaluation of an objective function over a list of points.

    SciPy's ODEPACK (zvode) uses Fortran static COMMON blocks and is NOT thread-safe
    across multiple threads in a single process. Therefore, true multiprocessing
    (fork on Linux/macOS, spawn on Windows) MUST be used rather than multi-threading.
    """
    global _PARALLEL_OBJECTIVE
    if max_workers is None:
        max_workers = mp.cpu_count()

    if max_workers <= 1 or len(points) <= 1:
        return [float(objective(p)) for p in points]

    available_methods = mp.get_all_start_methods()

    # 1. On Linux and macOS, 'fork' is fast and inherits memory via copy-on-write
    if "fork" in available_methods:
        try:
            _PARALLEL_OBJECTIVE = objective
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=max_workers) as pool:
                return pool.map(_parallel_worker_evaluate, points)
        except Exception:
            pass
        finally:
            _PARALLEL_OBJECTIVE = None

    # 2. On Windows (or when fork is unavailable), use 'spawn' process pool
    # with a top-level initializer to avoid thread-safety violations in zvode.
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max_workers, initializer=_parallel_worker_init, initargs=(objective,)) as pool:
            return pool.map(_parallel_worker_evaluate, points)
    except Exception:
        pass

    # 3. Fallback: sequential execution (safe against ODEPACK concurrency bugs)
    return [float(objective(p)) for p in points]


class LegendrePulseObjective:
    """Picklable objective callable for multi-process optimization across platforms."""

    def __init__(
        self,
        t_ns: np.ndarray,
        Omega: np.ndarray,
        basis: dict[str, Any],
        fidelity_fn: Callable[[np.ndarray, np.ndarray, float], float],
        T_ns: float,
    ) -> None:
        self.t_ns = t_ns
        self.Omega = Omega
        self.basis = basis
        self.fidelity_fn = fidelity_fn
        self.T_ns = T_ns

    def __call__(self, coefficients: np.ndarray) -> float:
        _, theta = reconstruct_eta_theta(self.t_ns, self.Omega, self.basis, coefficients)
        value = float(self.fidelity_fn(self.Omega, theta, self.T_ns))
        if not np.isfinite(value):
            raise ValueError("fidelity_fn returned a non-finite value.")
        return value


def finite_difference_gradient(
    objective: Callable[[np.ndarray], float],
    x: np.ndarray,
    relative_step: float = 1e-4,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """Central finite-difference gradient of a scalar objective with multi-core parallel evaluation."""
    x = np.asarray(x, dtype=float)
    dim = x.size
    if dim == 0:
        return np.zeros(0, dtype=float)

    h_list = [relative_step * max(1.0, abs(float(x[i]))) for i in range(dim)]
    eval_points = []
    for i in range(dim):
        h = h_list[i]
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        eval_points.append(xp)
        eval_points.append(xm)

    vals = parallel_evaluate_points(objective, eval_points, max_workers=max_workers)

    grad = np.zeros(dim, dtype=float)
    for i in range(dim):
        f_plus = vals[2 * i]
        f_minus = vals[2 * i + 1]
        grad[i] = (f_plus - f_minus) / (2.0 * h_list[i])
    return grad


def gradient_ascent(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 300,
    grad_tol: float = 1e-7,
    finite_difference_step: float = 1e-4,
    armijo: float = 1e-4,
    backtracking_factor: float = 0.5,
    min_step: float = 1e-8,
    max_workers: Optional[int] = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> OptimizationResult:
    """Maximize objective using finite-difference gradient ascent + line search.

    Monitors progress and computes finite-difference gradient in parallel using max CPU cores.
    Optionally invokes callback(iteration, x, fidelity) at iteration 0 and after every step.
    """
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("x0 must be a 1-D array.")
    if learning_rate <= 0.0 or max_iter < 1:
        raise ValueError("learning_rate must be positive and max_iter >= 1.")

    if max_workers is None:
        max_workers = mp.cpu_count()

    if verbose:
        print(f"\n[Optimizer] Starting gradient ascent (Workers: {max_workers}, Dim: {x.size}, MaxIter: {max_iter})")

    f = float(objective(x))
    history: list[dict[str, float]] = [
        {"iteration": 0.0, "fidelity": f, "gradient_norm": np.nan}
    ]

    if verbose:
        print(f"[Optimizer] Iteration 0 | Initial Fidelity = {f:.8f} | Coefficients = {x}")

    if callback is not None:
        try:
            callback(0, x, f)
        except Exception as e:
            if verbose:
                print(f"[Optimizer] Warning in callback at iteration 0: {e}")

    if x.size == 0:
        return OptimizationResult(x, f, 0, True, "No free coefficients (N=0).", history)

    for iteration in range(1, max_iter + 1):
        grad = finite_difference_gradient(
            objective,
            x,
            relative_step=finite_difference_step,
            max_workers=max_workers,
        )
        grad_norm = float(np.linalg.norm(grad))
        history[-1]["gradient_norm"] = grad_norm

        if not np.isfinite(grad_norm):
            msg = "Non-finite gradient encountered."
            if verbose:
                print(f"[Optimizer] Terminated: {msg}")
            return OptimizationResult(x, f, iteration - 1, False, msg, history)

        if grad_norm <= grad_tol:
            msg = "Gradient norm reached tolerance."
            if verbose:
                print(f"[Optimizer] Converged: {msg} (||grad|| = {grad_norm:.3e})")
            return OptimizationResult(x, f, iteration - 1, True, msg, history)

        # Normalize direction
        direction = grad / grad_norm
        directional_derivative = float(np.dot(grad, direction))
        step = learning_rate
        accepted = False

        while step >= min_step:
            x_trial = x + step * direction
            f_trial = float(objective(x_trial))
            # Accept if strictly improving or satisfies Armijo condition
            if np.isfinite(f_trial) and (
                f_trial >= (f + armijo * step * directional_derivative) or f_trial > f
            ):
                accepted = True
                break
            step *= backtracking_factor

        if not accepted:
            msg = "Line search could not find a further improving step."
            if verbose:
                print(f"[Optimizer] Terminated at iteration {iteration}: {msg}")
            return OptimizationResult(x, f, iteration - 1, True, msg, history)

        x = x_trial
        f = f_trial
        history.append(
            {
                "iteration": float(iteration),
                "fidelity": f,
                "gradient_norm": grad_norm,
            }
        )

        if verbose:
            print(
                f"[Optimizer] Iteration {iteration:3d} | "
                f"Fidelity = {f:.8f} | "
                f"||grad|| = {grad_norm:.3e} | "
                f"Step = {step:.3e} | "
                f"Coeffs = {np.round(x, 5)}"
            )

        if callback is not None:
            try:
                callback(iteration, x, f)
            except Exception as e:
                if verbose:
                    print(f"[Optimizer] Warning in callback at iteration {iteration}: {e}")

    msg = "Reached max_iter before convergence."
    if verbose:
        print(f"[Optimizer] {msg}")
    return OptimizationResult(x, f, max_iter, False, msg, history)


# ---------------------------------------------------------------------------
# Top-level optimization wrapper
# ---------------------------------------------------------------------------
def save_iteration_data(
    iteration: int,
    t_grid: np.ndarray,
    Omega: np.ndarray,
    theta: np.ndarray,
    fidelity: float,
    *,
    coefficients: Optional[np.ndarray] = None,
    output_dir: str = "optimization_history",
    system_cache: Optional[dict[str, Any]] = None,
) -> None:
    """Save pulse CSV, results TXT, and plots of pulse envelopes and populations for an iteration."""
    os.makedirs(output_dir, exist_ok=True)
    if system_cache is None:
        system_cache = get_stirap_system_cache()

    Omega_p = Omega * np.sin(theta)
    Omega_s = Omega * np.cos(theta)

    # 1. Save Pulse Envelopes to CSV
    Omega_p_mhz = Omega_p / (2.0 * np.pi * 1e-3)
    Omega_s_mhz = Omega_s / (2.0 * np.pi * 1e-3)
    Omega_mhz = Omega / (2.0 * np.pi * 1e-3)

    csv_path = os.path.join(output_dir, f"pulse_iter_{iteration:03d}.csv")
    pulse_table = np.column_stack(
        [t_grid, Omega, theta, Omega_s, Omega_p, Omega_mhz, Omega_s_mhz, Omega_p_mhz]
    )
    header = (
        "time_ns,Omega_rad_ns,theta_rad,Omega_s_rad_ns,Omega_p_rad_ns,"
        "Omega_mhz,Omega_s_mhz,Omega_p_mhz"
    )
    np.savetxt(csv_path, pulse_table, delimiter=",", header=header, comments="", fmt="%.8e")

    # 2. Simulate 3-level & other population dynamics
    N = system_cache["N"]
    P0 = qt.basis(N, 0).proj()
    P2 = qt.basis(N, 2).proj()
    P3 = qt.basis(N, 3).proj()
    P_other = qt.qeye(N) - P0 - P2 - P3

    sim_res = simulate_stirap_arbitrary_pulse(
        t_grid=t_grid,
        Omega_p=Omega_p,
        Omega_s=Omega_s,
        system_cache=system_cache,
        e_ops=[P0, P3, P2, P_other],
    )
    P0_t, P3_t, P2_t, Pother_t = [np.real(np.asarray(x)) for x in sim_res.expect]

    final_P0 = float(P0_t[-1])
    final_P2 = float(P2_t[-1])
    final_P3 = float(P3_t[-1])
    final_Pother = float(Pother_t[-1])
    p3_max = float(np.max(P3_t))

    # 3. Save Results Summary to TXT
    txt_path = os.path.join(output_dir, f"results_iter_{iteration:03d}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"STIRAP ITERATION {iteration:03d} RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Iteration:     {iteration}\n")
        f.write(f"Fidelity (P2): {final_P2:.8f}\n")
        f.write(f"Final P0:      {final_P0:.8f}\n")
        f.write(f"Final P2:      {final_P2:.8f}\n")
        f.write(f"Final P3:      {final_P3:.8f}\n")
        f.write(f"Final P_other: {final_Pother:.8f}\n")
        f.write(f"P3_max:        {p3_max:.8f}\n")
        f.write(f"theta(0):      {theta[0]:.8f} rad\n")
        f.write(f"theta(T):      {theta[-1]:.8f} rad\n")
        if coefficients is not None:
            f.write(f"Coefficients:  {np.array2string(coefficients, precision=8, separator=', ')}\n")
        f.write("=" * 60 + "\n")

    # 4. Plot Pulse Envelopes
    pulse_img_path = os.path.join(output_dir, f"pulse_iter_{iteration:03d}.png")
    fig1, ax1 = plt.subplots(figsize=(7.2, 2.8))
    ax1.plot(t_grid, Omega_s_mhz, "--", label=r"$\Omega_s$")
    ax1.plot(t_grid, Omega_p_mhz, "-", label=r"$\Omega_p$")
    ax1.set_title(f"Iteration {iteration} Rabi Envelopes (Fidelity = {fidelity:.6f})")
    ax1.set_ylabel("Rabi frequency [MHz]")
    ax1.set_xlabel("Time [ns]")
    ax1.set_xlim(float(t_grid[0]), float(t_grid[-1]))
    peak_val = max(float(np.max(Omega_p_mhz)), float(np.max(Omega_s_mhz)))
    ax1.set_ylim(0.0, peak_val * 1.1 + 1.0)
    ax1.legend(loc="upper right")
    ax1.grid(False)
    fig1.tight_layout()
    fig1.savefig(pulse_img_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 5. Plot 3-level & other Populations
    pop_img_path = os.path.join(output_dir, f"populations_iter_{iteration:03d}.png")
    fig2, ax2 = plt.subplots(figsize=(7.2, 2.8))
    ax2.plot(t_grid, P0_t, label=r"$p_{|0\rangle}$")
    ax2.plot(t_grid, P3_t, label=r"$p_{|3\rangle}$")
    ax2.plot(t_grid, P2_t, label=r"$p_{|2\rangle}$")
    if np.max(Pother_t) > 1e-4:
        ax2.plot(t_grid, Pother_t, ":", label=r"$p_{\mathrm{other}}$")
    ax2.set_title(f"Iteration {iteration} Populations (P2={final_P2:.6f}, P3_max={p3_max:.6f}, P_other={final_Pother:.6f})")
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("Population")
    ax2.set_xlim(float(t_grid[0]), float(t_grid[-1]))
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="center right")
    ax2.grid(False)
    fig2.tight_layout()
    fig2.savefig(pop_img_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(
        f"  [Saved Iteration {iteration}] CSV: {csv_path} | "
        f"TXT: {txt_path} | Plots: {pulse_img_path}, {pop_img_path}"
    )


def optimize_flat_top_pulse(
    fidelity_fn: Callable[[np.ndarray, np.ndarray, float], float],
    *,
    Omega_initial: Optional[np.ndarray] = None,
    T_ns: float = T_NS,
    omega0_over_2pi_mhz: float = OMEGA0_OVER_2PI_MHZ,
    max_degree: int = 6,
    n_time: int = 2001,
    ramp_calibration: Callable[[float], tuple[float, float]] = power_law_ramp_calibration,
    optimizer: Callable[..., OptimizationResult] = gradient_ascent,
    optimizer_kwargs: Optional[dict[str, Any]] = None,
    initial_coefficients: Optional[np.ndarray] = None,
    save_step: int = 1,
    output_dir: str = "optimization_history",
    system_cache: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build pulse/basis, optimize a_1,...,a_N, and return all useful arrays.

    Parameters
    ----------
    fidelity_fn:
        User's function: fidelity_fn(Omega_array, theta_array, T_ns) -> scalar.
        The two arrays have shape (n_time,) and correspond to t_ns.
    Omega_initial:
        Optional precomputed Omega array (shape n_time). If None, generates
        the default flat-top Gaussian envelope.
    max_degree:
        Highest Legendre seed degree. There are max_degree free coefficients
        a_1,...,a_N; a_0 is fixed by the integral constraint.
    ramp_calibration:
        Function of Omega0/(2*pi) in MHz returning (rise_time_ns, sigma_ns).
    optimizer:
        Optimizer with interface optimizer(objective, x0, **optimizer_kwargs).
    save_step:
        Save pulse CSV and pulse/population plots every `save_step` iterations (default: 1).
    output_dir:
        Directory path where iteration CSV and plots are saved.
    """
    if n_time < max_degree + 2:
        raise ValueError("n_time must be at least max_degree + 2.")
    t_ns = np.linspace(0.0, T_ns, n_time)
    omega0_rad_ns = 2.0 * np.pi * (omega0_over_2pi_mhz * 1e-3)
    rise_time_ns, sigma_ns = ramp_calibration(omega0_over_2pi_mhz)

    if Omega_initial is not None:
        Omega = np.asarray(Omega_initial, dtype=float).copy()
        if Omega.shape != t_ns.shape:
            raise ValueError(f"Omega_initial shape {Omega.shape} must match t_ns shape {t_ns.shape}.")
    else:
        Omega = flat_top_gaussian_envelope(
            t_ns=t_ns,
            T_ns=T_ns,
            omega0_rad_ns=omega0_rad_ns,
            rise_time_ns=rise_time_ns,
            sigma_ns=sigma_ns,
        )
    basis = build_weighted_legendre_basis(t_ns, Omega, max_degree)

    if initial_coefficients is None:
        x0 = np.zeros(max_degree, dtype=float)
    else:
        x0 = np.asarray(initial_coefficients, dtype=float)
        if x0.shape != (max_degree,):
            raise ValueError(
                f"initial_coefficients must have shape ({max_degree},), got {x0.shape}."
            )

    objective = LegendrePulseObjective(
        t_ns=t_ns,
        Omega=Omega,
        basis=basis,
        fidelity_fn=fidelity_fn,
        T_ns=T_ns,
    )

    def iteration_callback(iter_num: int, coeffs: np.ndarray, current_fid: float) -> None:
        if save_step > 0 and (iter_num % save_step == 0):
            _, current_theta = reconstruct_eta_theta(t_ns, Omega, basis, coeffs)
            save_iteration_data(
                iteration=iter_num,
                t_grid=t_ns,
                Omega=Omega,
                theta=current_theta,
                fidelity=current_fid,
                coefficients=coeffs,
                output_dir=output_dir,
                system_cache=system_cache,
            )

    options = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    if "callback" not in options:
        options["callback"] = iteration_callback

    opt_result = optimizer(objective, x0, **options)
    eta_opt, theta_opt = reconstruct_eta_theta(t_ns, Omega, basis, opt_result.x)

    q = basis["quadrature_weights"]
    endpoint_constraint = float(np.sum(q * eta_opt * Omega))
    endpoint_angle = float(theta_opt[-1])

    return {
        "t_ns": t_ns,
        "Omega": Omega,
        "theta": theta_opt,
        "eta": eta_opt,
        "coefficients": opt_result.x.copy(),
        "fidelity": opt_result.fun,
        "optimization": opt_result,
        "basis": basis["Phi"],
        "basis_data": basis,
        "W": basis["W"],
        "omega0_rad_ns": omega0_rad_ns,
        "rise_time_ns": rise_time_ns,
        "sigma_ns": sigma_ns,
        "endpoint_constraint_value": endpoint_constraint,
        "theta_T": endpoint_angle,
    }


def optimize_stirap_with_iterative_omega(
    fidelity_fn: Callable[[np.ndarray, np.ndarray, float], float],
    *,
    T_ns: float = T_NS,
    omega0_over_2pi_mhz: float = OMEGA0_OVER_2PI_MHZ,
    max_degree: int = 6,
    n_time: int = 2001,
    n_outer_iterations: int = 10,
    ramp_calibration: Callable[[float], tuple[float, float]] = power_law_ramp_calibration,
    optimizer: Callable[..., OptimizationResult] = gradient_ascent,
    stage_configs: Optional[list[dict[str, Any]]] = None,
    save_step: int = 1,
    output_dir: str = "optimization_history",
    system_cache: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Iteratively optimize theta and update Omega self-consistently over multiple rounds.

    Formula for updating Omega:
        Omega_new = Omega * min((n03 / n32) / sin(theta), 1 / cos(theta))

    Default 10-round schedule:
      - Rounds 1-3 (outer_iter 0, 1, 2): Coarse tuning (max_iter=15, grad_tol=1e-4) to settle Omega shape.
      - Rounds 4-6 (outer_iter 3, 4, 5): Medium tuning (max_iter=30, grad_tol=1e-5).
      - Rounds 7-10 (outer_iter 6, 7, 8, 9): Fine tuning (max_iter=50, grad_tol=1e-7) to maximize fidelity.

    Each outer round resets initial coefficients to all zeros and rebuilds the
    weighted Legendre basis on the newly modified Omega envelope.
    """
    if system_cache is None:
        system_cache = get_stirap_system_cache()

    n03 = float(system_cache["n03"])
    n32 = float(system_cache["n32"])

    # Configure stages for each outer iteration if not provided
    if stage_configs is None:
        stage_configs = []
        for r in range(n_outer_iterations):
            if r < 3:
                # Rounds 1-3: coarse tuning
                stage_configs.append({"max_iter": 15, "grad_tol": 1e-4, "learning_rate": 0.1})
            elif r < 6:
                # Rounds 4-6: medium tuning
                stage_configs.append({"max_iter": 30, "grad_tol": 1e-5, "learning_rate": 0.1})
            else:
                # Rounds 7-10: fine tuning
                stage_configs.append({"max_iter": 50, "grad_tol": 1e-7, "learning_rate": 0.1})
    elif len(stage_configs) < n_outer_iterations:
        last_config = stage_configs[-1]
        for _ in range(len(stage_configs), n_outer_iterations):
            stage_configs.append(dict(last_config))

    t_ns = np.linspace(0.0, T_ns, n_time)
    omega0_rad_ns = 2.0 * np.pi * (omega0_over_2pi_mhz * 1e-3)
    rise_time_ns, sigma_ns = ramp_calibration(omega0_over_2pi_mhz)

    Omega_base = flat_top_gaussian_envelope(
        t_ns=t_ns,
        T_ns=T_ns,
        omega0_rad_ns=omega0_rad_ns,
        rise_time_ns=rise_time_ns,
        sigma_ns=sigma_ns,
    )
    current_Omega = Omega_base.copy()
    current_coeffs: Optional[np.ndarray] = None
    prev_eta: Optional[np.ndarray] = None

    outer_results: list[dict[str, Any]] = []

    print("\n" + "=" * 75)
    print(f"STARTING SELF-CONSISTENT OMEGA-THETA ITERATION ({n_outer_iterations} ROUNDS)")
    print(f"Dipole matrix elements: n03 = {n03:.6f}, n32 = {n32:.6f}, ratio = {n03/n32:.6f}")
    print("=" * 75)

    for outer_idx in range(n_outer_iterations):
        stage_name = (
            "Coarse tuning" if outer_idx < 3 else ("Medium tuning" if outer_idx < 6 else "Fine tuning")
        )
        stage_kwargs = dict(stage_configs[outer_idx])
        round_dir = os.path.join(output_dir, f"outer_round_{outer_idx + 1:02d}")

        print(f"\n>>> [Round {outer_idx + 1:02d}/{n_outer_iterations:02d}] {stage_name} "
              f"(max_iter={stage_kwargs.get('max_iter', 50)}, grad_tol={stage_kwargs.get('grad_tol', 1e-7)})")

        # In round 0, start from zeros.
        # In later rounds, warm-start with coefficients projected from previous round's optimal eta.
        if outer_idx == 0 or prev_eta is None:
            init_coeffs = np.zeros(max_degree, dtype=float)
        else:
            temp_basis = build_weighted_legendre_basis(t_ns, current_Omega, max_degree)
            init_coeffs = project_eta_to_basis(prev_eta, temp_basis)
            print(f">>> [Warm-start] Projected coefficients from previous round: {np.round(init_coeffs, 5)}")

        round_result = optimize_flat_top_pulse(
            fidelity_fn=fidelity_fn,
            Omega_initial=current_Omega,
            T_ns=T_ns,
            omega0_over_2pi_mhz=omega0_over_2pi_mhz,
            max_degree=max_degree,
            n_time=n_time,
            ramp_calibration=ramp_calibration,
            optimizer=optimizer,
            optimizer_kwargs=stage_kwargs,
            initial_coefficients=init_coeffs,
            save_step=save_step,
            output_dir=round_dir,
            system_cache=system_cache,
        )

        current_theta = round_result["theta"]
        prev_eta = round_result["eta"]
        current_coeffs = round_result["coefficients"]

        outer_results.append({
            "round": outer_idx + 1,
            "stage": stage_name,
            "fidelity": round_result["fidelity"],
            "Omega": current_Omega.copy(),
            "theta": current_theta.copy(),
            "coefficients": current_coeffs.copy(),
            "result": round_result,
        })

        print(f">>> [Round {outer_idx + 1:02d} Completed] Fidelity = {round_result['fidelity']:.8f}")

        # Update Omega for the next round by applying the correction formula to Omega_base
        if outer_idx < n_outer_iterations - 1:
            new_Omega = update_omega_from_theta(
                Omega=Omega_base,
                theta=current_theta,
                n03=n03,
                n32=n32,
            )
            # Log change in Omega envelope compared to previous round
            max_diff = float(np.max(np.abs(new_Omega - current_Omega)))
            rel_diff = max_diff / float(np.max(current_Omega))
            print(f">>> [Omega Updated from Base] max|Omega_new - Omega_old| = {max_diff:.4e} rad/ns (relative {rel_diff*100:.2f}%)")
            current_Omega = new_Omega

    last_res = outer_results[-1]["result"]
    return {
        "t_ns": t_ns,
        "Omega": current_Omega,
        "theta": last_res["theta"],
        "eta": last_res["eta"],
        "coefficients": last_res["coefficients"],
        "fidelity": last_res["fidelity"],
        "optimization": last_res["optimization"],
        "basis": last_res["basis"],
        "basis_data": last_res["basis_data"],
        "W": last_res["W"],
        "omega0_rad_ns": omega0_rad_ns,
        "rise_time_ns": rise_time_ns,
        "sigma_ns": sigma_ns,
        "endpoint_constraint_value": last_res["endpoint_constraint_value"],
        "theta_T": last_res["theta_T"],
        "outer_history": outer_results,
    }


# ---------------------------------------------------------------------------
# Plotting and Reporting Utilities
# ---------------------------------------------------------------------------
def plot_stirap_results(
    tlist: np.ndarray,
    Omega_p: np.ndarray,
    Omega_s: np.ndarray,
    P0_t: np.ndarray,
    P3_t: np.ndarray,
    P2_t: np.ndarray,
    Pother_t: Optional[np.ndarray] = None,
    env_type: str = "transmission_line",
    pulse_filename: Optional[str] = None,
    pop_filename: Optional[str] = None,
) -> None:
    """Plot and save Rabi frequency envelopes and population transfer dynamics.

    Matches the publication-quality plotting format of run_stirap.py.
    """
    if pulse_filename is None:
        pulse_filename = f"stirap_rabi_pulses_{env_type}.png"
    if pop_filename is None:
        pop_filename = f"stirap_populations_{env_type}.png"

    # Convert rad/ns to MHz
    Omega_p_mhz = Omega_p / (2.0 * np.pi * 1e-3)
    Omega_s_mhz = Omega_s / (2.0 * np.pi * 1e-3)
    peak_val = max(float(np.max(Omega_p_mhz)), float(np.max(Omega_s_mhz)))

    # Plot 1: Rabi-frequency envelopes
    fig1, ax1 = plt.subplots(figsize=(7.2, 2.8))
    ax1.plot(tlist, Omega_s_mhz, "--", label=r"$\Omega_s$")
    ax1.plot(tlist, Omega_p_mhz, "-", label=r"$\Omega_p$")
    ax1.set_ylabel("Rabi frequency [MHz]")
    ax1.set_xlabel("Time [ns]")
    ax1.set_xlim(float(tlist[0]), float(tlist[-1]))
    ax1.set_ylim(0.0, peak_val * 1.1 + 1.0)
    ax1.legend(loc="upper right")
    ax1.grid(False)
    fig1.tight_layout()
    fig1.savefig(pulse_filename, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Population transfer
    fig2, ax2 = plt.subplots(figsize=(7.2, 2.8))
    ax2.plot(tlist, P0_t, label=r"$p_{|0\rangle}$")
    ax2.plot(tlist, P3_t, label=r"$p_{|3\rangle}$")
    ax2.plot(tlist, P2_t, label=r"$p_{|2\rangle}$")
    if Pother_t is not None and np.max(Pother_t) > 1e-4:
        ax2.plot(tlist, Pother_t, ":", label=r"$p_{\mathrm{other}}$")
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylabel("Population")
    ax2.set_xlim(float(tlist[0]), float(tlist[-1]))
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="center right")
    ax2.grid(False)
    fig2.tight_layout()
    fig2.savefig(pop_filename, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nSaved figures:\n  {pulse_filename}\n  {pop_filename}")


# ---------------------------------------------------------------------------
# Global Picklable STIRAP Fidelity Objective
# ---------------------------------------------------------------------------
class StirapFidelityObjective:
    """Picklable STIRAP fidelity evaluator for cross-platform multiprocessing (spawn)."""

    def __init__(self, env_type: str = "transmission_line", T_kelvin: float = 0.030) -> None:
        self.env_type = env_type
        self.T_kelvin = T_kelvin

    def __call__(self, Omega: np.ndarray, theta: np.ndarray, T: float) -> float:
        cache = get_stirap_system_cache(env_type=self.env_type, T_kelvin=self.T_kelvin)
        return calculate_stirap_fidelity(
            Omega_array=Omega,
            theta_array=theta,
            T_ns=T,
            system_cache=cache,
        )


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("0-3-2 STIRAP WEIGHTED LEGENDRE PULSE OPTIMIZATION")
    print("=" * 70)
    #env_type = "transmission_line"
    env_type = "multi_stage_bpf"
    # 1. Initialize physical system
    system_cache = get_stirap_system_cache(env_type=env_type, T_kelvin=0.030)
    print(f"System initialized with N = {system_cache['N']} levels.")
    print(f"Number of collapse operators = {len(system_cache['c_ops'])}")

    # 2. Setup picklable fidelity objective for STIRAP
    stirap_fidelity_objective = StirapFidelityObjective(env_type=env_type, T_kelvin=0.030)


    # 3. Run self-consistent iterative optimization (10 rounds)
    max_cores = mp.cpu_count()
    save_step = 1  # Save CSV and plots every save_step iterations (1 = every iteration)
    history_dir = "optimization_history"
    n_outer_iterations = 10
    print(f"Starting 10-round iterative optimization using {max_cores} CPU cores...")
    print(f"Rounds 1-3: coarse tuning | Rounds 4-6: medium tuning | Rounds 7-10: fine tuning")
    print(f"Saving pulse CSV, pulse plot, and 3-level population plot every {save_step} iteration(s) to '{history_dir}'...")

    # Stage configs for 10 rounds
    stage_configs = []
    for r in range(n_outer_iterations):
        if r < 3:
            stage_configs.append({"max_iter": 15, "grad_tol": 1e-4, "learning_rate": 0.1, "max_workers": max_cores, "verbose": True})
        elif r < 6:
            stage_configs.append({"max_iter": 30, "grad_tol": 1e-5, "learning_rate": 0.1, "max_workers": max_cores, "verbose": True})
        else:
            stage_configs.append({"max_iter": 50, "grad_tol": 1e-7, "learning_rate": 0.1, "max_workers": max_cores, "verbose": True})

    result = optimize_stirap_with_iterative_omega(
        fidelity_fn=stirap_fidelity_objective,
        T_ns=T_NS,
        omega0_over_2pi_mhz=OMEGA0_OVER_2PI_MHZ,
        max_degree=6,
        n_time=2001,
        n_outer_iterations=n_outer_iterations,
        stage_configs=stage_configs,
        save_step=save_step,
        output_dir=history_dir,
        system_cache=system_cache,
    )

    t_grid = result["t_ns"]
    Omega_opt = result["Omega"]
    theta_opt = result["theta"]

    Omega_p_opt = Omega_opt * np.sin(theta_opt)
    Omega_s_opt = Omega_opt * np.cos(theta_opt)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Omega0/(2pi)             = {OMEGA0_OVER_2PI_MHZ:.3f} MHz")
    print(f"Pulse duration T         = {T_NS:.1f} ns")
    print(f"Final Optimized Fidelity = {result['fidelity']:.6f}")
    print(f"Optimal Coefficients     = {np.round(result['coefficients'], 6)}")
    print(f"theta(0)                 = {theta_opt[0]:.6f} rad")
    print(f"theta(T)                 = {result['theta_T']:.6f} rad (expected pi/2 = {np.pi/2:.6f})")
    print(f"Endpoint constraint      = {result['endpoint_constraint_value']:.8f}")

    # 4. Simulate full time-evolution dynamics with optimized pulses
    print("\nSimulating full time-dependent population dynamics...")
    P0 = qt.basis(system_cache["N"], 0).proj()
    P2 = qt.basis(system_cache["N"], 2).proj()
    P3 = qt.basis(system_cache["N"], 3).proj()
    P_other = qt.qeye(system_cache["N"]) - P0 - P2 - P3

    sim_res = simulate_stirap_arbitrary_pulse(
        t_grid=t_grid,
        Omega_p=Omega_p_opt,
        Omega_s=Omega_s_opt,
        system_cache=system_cache,
        e_ops=[P0, P3, P2, P_other],
    )

    P0_t, P3_t, P2_t, Pother_t = [np.real(np.asarray(x)) for x in sim_res.expect]

    print("\nSTIRAP Simulation Results:")
    print(f"Final P0      = {P0_t[-1]:.6f}")
    print(f"Final P3      = {P3_t[-1]:.6f}")
    print(f"Final P2      = {P2_t[-1]:.6f}  (Target Fidelity)")
    print(f"Final P_other = {Pother_t[-1]:.6f}")
    print(f"Maximum P3    = {np.max(P3_t):.6f}")

    # 5. Generate and save plots
    plot_stirap_results(
        tlist=t_grid,
        Omega_p=Omega_p_opt,
        Omega_s=Omega_s_opt,
        P0_t=P0_t,
        P3_t=P3_t,
        P2_t=P2_t,
        Pother_t=Pother_t,
        env_type=system_cache["env_type"],
    )
