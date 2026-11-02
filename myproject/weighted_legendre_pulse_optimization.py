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

import numpy as np
from numpy.polynomial.legendre import legvander


# ---------------------------------------------------------------------------
# User-provided pulse settings
# ---------------------------------------------------------------------------
T_NS = 300.0
OMEGA0_OVER_2PI_MHZ = 20.0
OMEGA0_RAD_NS = 2.0 * np.pi * (OMEGA0_OVER_2PI_MHZ * 1e-3)


# ---------------------------------------------------------------------------
# Instrument ramp calibration
# ---------------------------------------------------------------------------
def power_law_ramp_calibration(
    omega0_over_2pi_mhz: float,
    *,
    reference_omega_mhz: float = 20.0,
    reference_rise_time_ns: float = 20.0,
    reference_sigma_ns: float = 5.0,
    rise_time_exponent: float = 0.5,
    sigma_exponent: float = 0.5,
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


def finite_difference_gradient(
    objective: Callable[[np.ndarray], float],
    x: np.ndarray,
    relative_step: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference gradient of a scalar objective."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        h = relative_step * max(1.0, abs(float(x[i])))
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (float(objective(xp)) - float(objective(xm))) / (2.0 * h)
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
) -> OptimizationResult:
    """Maximize objective using finite-difference gradient ascent + line search.

    This is gradient descent applied to -objective, written as ascent because
    the quantity of interest (fidelity) is to be maximized.
    """
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("x0 must be a 1-D array.")
    if learning_rate <= 0.0 or max_iter < 1:
        raise ValueError("learning_rate must be positive and max_iter >= 1.")

    f = float(objective(x))
    history: list[dict[str, float]] = [
        {"iteration": 0.0, "fidelity": f, "gradient_norm": np.nan}
    ]

    if x.size == 0:
        return OptimizationResult(x, f, 0, True, "No free coefficients (N=0).", history)

    for iteration in range(1, max_iter + 1):
        grad = finite_difference_gradient(objective, x, finite_difference_step)
        grad_norm = float(np.linalg.norm(grad))
        history[-1]["gradient_norm"] = grad_norm

        if not np.isfinite(grad_norm):
            return OptimizationResult(
                x, f, iteration - 1, False, "Non-finite gradient encountered.", history
            )
        if grad_norm <= grad_tol:
            return OptimizationResult(
                x, f, iteration - 1, True, "Gradient norm reached tolerance.", history
            )

        # Normalize the direction so learning_rate is a step length in
        # coefficient space rather than being strongly dependent on gradient scale.
        direction = grad / grad_norm
        directional_derivative = float(np.dot(grad, direction))
        step = learning_rate
        accepted = False

        while step >= min_step:
            x_trial = x + step * direction
            f_trial = float(objective(x_trial))
            if np.isfinite(f_trial) and f_trial >= (
                f + armijo * step * directional_derivative
            ):
                accepted = True
                break
            step *= backtracking_factor

        if not accepted:
            return OptimizationResult(
                x, f, iteration - 1, True,
                "Line search could not find a further improving step.", history
            )

        x = x_trial
        f = f_trial
        history.append(
            {
                "iteration": float(iteration),
                "fidelity": f,
                "gradient_norm": grad_norm,
            }
        )

    return OptimizationResult(
        x, f, max_iter, False, "Reached max_iter before convergence.", history
    )


# ---------------------------------------------------------------------------
# Top-level optimization wrapper
# ---------------------------------------------------------------------------
def optimize_flat_top_pulse(
    fidelity_fn: Callable[[np.ndarray, np.ndarray, float], float],
    *,
    T_ns: float = T_NS,
    omega0_over_2pi_mhz: float = OMEGA0_OVER_2PI_MHZ,
    max_degree: int = 6,
    n_time: int = 2001,
    ramp_calibration: Callable[[float], tuple[float, float]] = power_law_ramp_calibration,
    optimizer: Callable[..., OptimizationResult] = gradient_ascent,
    optimizer_kwargs: Optional[dict[str, Any]] = None,
    initial_coefficients: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Build pulse/basis, optimize a_1,...,a_N, and return all useful arrays.

    Parameters
    ----------
    fidelity_fn:
        User's function: fidelity_fn(Omega_array, theta_array, T_ns) -> scalar.
        The two arrays have shape (n_time,) and correspond to t_ns.
    max_degree:
        Highest Legendre seed degree. There are max_degree free coefficients
        a_1,...,a_N; a_0 is fixed by the integral constraint.
    ramp_calibration:
        Function of Omega0/(2*pi) in MHz returning (rise_time_ns, sigma_ns).
    optimizer:
        Optimizer with interface optimizer(objective, x0, **optimizer_kwargs).
        Replace gradient_ascent with a PSO or other optimizer adapter later.
    """
    if n_time < max_degree + 2:
        raise ValueError("n_time must be at least max_degree + 2.")
    t_ns = np.linspace(0.0, T_ns, n_time)
    omega0_rad_ns = 2.0 * np.pi * (omega0_over_2pi_mhz * 1e-3)
    rise_time_ns, sigma_ns = ramp_calibration(omega0_over_2pi_mhz)

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

    def objective(coefficients: np.ndarray) -> float:
        _, theta = reconstruct_eta_theta(t_ns, Omega, basis, coefficients)
        value = float(fidelity_fn(Omega, theta, T_ns))
        if not np.isfinite(value):
            raise ValueError("fidelity_fn returned a non-finite value.")
        return value

    options = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
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


# ---------------------------------------------------------------------------
# Example usage (replace example_fidelity with your actual fidelity function)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def example_fidelity(Omega: np.ndarray, theta: np.ndarray, T: float) -> float:
        """A dummy objective for smoke-testing only; not a physics fidelity."""
        target = 0.5 * np.pi * np.linspace(0.0, 1.0, theta.size)
        return float(-np.mean((theta - target) ** 2))

    result = optimize_flat_top_pulse(
        fidelity_fn=example_fidelity,
        T_ns=T_NS,
        omega0_over_2pi_mhz=OMEGA0_OVER_2PI_MHZ,
        max_degree=6,
        n_time=2001,
        optimizer_kwargs={"learning_rate": 0.2, "max_iter": 100},
    )

    print(f"Omega0/(2pi) = {OMEGA0_OVER_2PI_MHZ:.3f} MHz")
    print(f"Omega0 = {result['omega0_rad_ns']:.8f} rad/ns")
    print(f"T = {T_NS:.1f} ns")
    print(
        "Gaussian ramp: "
        f"rise_time={result['rise_time_ns']:.3f} ns, "
        f"sigma={result['sigma_ns']:.3f} ns"
    )
    print(f"W = {result['W']:.8f}")
    print(f"Optimized fidelity (dummy) = {result['fidelity']:.10f}")
    print(f"Coefficients a_1,...,a_N = {result['coefficients']}")
    print(f"Integral constraint = {result['endpoint_constraint_value']:.12f}")
    print(f"theta(T) = {result['theta_T']:.12f} rad")
    print(
        "Weighted orthogonality max error = "
        f"{result['basis_data']['orthogonality_error']:.3e}"
    )
