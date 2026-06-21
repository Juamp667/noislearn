"""Noise-score filtering with optional adaptive thresholding.

The adaptive branch fits two beta-shaped score populations on the interval
[0, 1] and uses their intersection as the filtering threshold. The rational
branch estimates a density over ``noise_score_`` and fits a flexible rational
surrogate to recover an internal valley threshold when the fitted shape really
supports one. The filter can also train a nested ``noise_filter`` on demand and
can threshold by an explicit quantile of the score distribution.
"""

import math
import warnings

import numpy as np

try:
    from imblearn.base import BaseSampler
except Exception:  # pragma: no cover - optional dependency fallback.
    class BaseSampler:  # type: ignore[too-many-ancestors]
        """Minimal fallback used when imbalanced-learn is unavailable."""

        def __init__(self, sampling_strategy="auto"):
            self.sampling_strategy = sampling_strategy

try:
    from scipy.optimize import OptimizeWarning, curve_fit
except Exception:  # pragma: no cover - SciPy is expected but optional.
    OptimizeWarning = Warning
    curve_fit = None


_BETA_EPS = 1e-12
_ADAPTATIVE_GRID_SIZE = 4096
_ADAPTATIVE_MAX_ITER = 50
_ADAPTATIVE_TOL = 1e-4
_RATIONAL_DENSITY_EPS = 1e-8
_RATIONAL_GRID_SIZE = 512
_RATIONAL_HIST_MIN_BINS = 5
_RATIONAL_HIST_MAX_BINS = 64
_RATIONAL_MARGIN = 0.05
_RATIONAL_MIN_SPREAD = 1e-4
_RATIONAL_LOCAL_EPS = 1e-3
_RATIONAL_SEPARABILITY_MIN = 0.10


def _as_1d_array(values, *, name):
    """Convert ``values`` to a strict 1D numpy array."""

    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1)
    raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")


def _beta_log_pdf(x, alpha, beta):
    """Evaluate the log-PDF of a beta distribution on an open interval."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta parameters must be positive.")
    x = np.asarray(x, dtype=float)
    log_norm = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log1p(-x) - log_norm


def _fit_beta_moments(values):
    """Fit a beta distribution from sample moments.

    Exact 0 and 1 scores are allowed by the public API. They are clipped only
    for the internal moment estimates so that the beta fit remains numerically
    stable on the open interval.
    """

    raw = np.asarray(values, dtype=float).ravel()
    if raw.size == 0:
        raise ValueError("Cannot fit a beta distribution on an empty sample.")

    # A singleton has no variance estimate. Use a broad beta centered on the
    # observation, but still respect the boundary cases when the value is 0 or 1.
    if raw.size == 1:
        value = float(raw[0])
        if value <= _BETA_EPS:
            return _BETA_EPS, 1.0
        if value >= 1.0 - _BETA_EPS:
            return 1.0, _BETA_EPS
        common = 2.0
        return max(value * common, _BETA_EPS), max((1.0 - value) * common, _BETA_EPS)

    values = np.clip(raw, _BETA_EPS, 1.0 - _BETA_EPS)
    mean = float(np.mean(values))
    if mean <= _BETA_EPS:
        return _BETA_EPS, 1.0
    if mean >= 1.0 - _BETA_EPS:
        return 1.0, _BETA_EPS

    var = float(np.var(values))
    max_var = mean * (1.0 - mean)
    if max_var <= _BETA_EPS:
        return 1.0, 1.0

    # Moment matching only works when the variance is below the beta maximum.
    # The clamp keeps the estimate inside the feasible region.
    var = min(max(var, _BETA_EPS), max_var * (1.0 - 1e-6))
    common = max_var / var - 1.0
    if not np.isfinite(common) or common <= 0.0:
        common = 1.0

    alpha = max(mean * common, _BETA_EPS)
    beta = max((1.0 - mean) * common, _BETA_EPS)
    return alpha, beta


def _interpolated_roots(grid, diff):
    """Return the approximate roots of ``diff`` on ``grid``.

    The helper looks for sign changes and refines them with linear
    interpolation. Exact zeros are also preserved, which is useful when both
    fitted densities are nearly identical.
    """

    roots = []
    exact = np.isclose(diff, 0.0, atol=1e-12, rtol=0.0)
    if np.any(exact):
        roots.extend(grid[exact].tolist())

    change_idx = np.where(diff[:-1] * diff[1:] < 0.0)[0]
    for idx in change_idx:
        x0 = float(grid[idx])
        x1 = float(grid[idx + 1])
        y0 = float(diff[idx])
        y1 = float(diff[idx + 1])
        roots.append(x0 - y0 * (x1 - x0) / (y1 - y0))

    if not roots:
        return np.array([], dtype=float)
    return np.unique(np.asarray(roots, dtype=float))


def _select_threshold_candidate(grid, diff, mean_left, mean_right):
    """Pick the crossing that best separates the two fitted components."""

    roots = _interpolated_roots(grid, diff)
    midpoint = 0.5 * (mean_left + mean_right)
    if roots.size == 0:
        return float(np.clip(midpoint, 0.0, 1.0))

    low = min(mean_left, mean_right)
    high = max(mean_left, mean_right)
    in_span = roots[(roots >= low) & (roots <= high)]
    candidates = in_span if in_span.size else roots
    return float(candidates[np.argmin(np.abs(candidates - midpoint))])


def _integrate_trapezoid(values, grid):
    """Integrate ``values`` over ``grid`` with a NumPy-version-safe helper."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, grid))
    return float(np.trapz(values, grid))


def _rational_quadratic_linear(x, a, b, c, d):
    """Evaluate ``(a*x**2 + b*x + c) / (d*x + 1)`` on ``x``.

    The denominator is kept explicit so that the caller can validate it before
    accepting the fitted surrogate as a threshold model.
    """

    x = np.asarray(x, dtype=float)
    denominator = d * x + 1.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return (a * x**2 + b * x + c) / denominator


def _smooth_density(values, window_size=5):
    """Lightly smooth a 1D density estimate with an edge-padded moving average."""

    density = np.asarray(values, dtype=float).ravel()
    if density.size < 3:
        return density

    window_size = int(window_size)
    if window_size < 3:
        return density
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, density.size if density.size % 2 == 1 else density.size - 1)
    if window_size < 3:
        return density

    kernel = np.ones(window_size, dtype=float) / float(window_size)
    padded = np.pad(density, (window_size // 2,), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _estimate_histogram_density(scores):
    """Estimate a density on ``[0, 1]`` from a clipped histogram."""

    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size < 5:
        raise ValueError("density estimation requires at least 5 scores.")

    n_bins = int(np.clip(np.ceil(np.sqrt(scores.size) * 2.0), _RATIONAL_HIST_MIN_BINS, _RATIONAL_HIST_MAX_BINS))
    density, edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0), density=True)
    counts, _ = np.histogram(scores, bins=n_bins, range=(0.0, 1.0), density=False)
    grid = 0.5 * (edges[:-1] + edges[1:])

    density = np.asarray(density, dtype=float) + _RATIONAL_DENSITY_EPS
    density = _smooth_density(density, window_size=min(5, n_bins))
    density = np.maximum(density, _RATIONAL_DENSITY_EPS)

    area = _integrate_trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("density estimation failed.")
    density = density / area

    return grid, density, counts.astype(float), n_bins


def _initial_rational_parameters(grid, density, counts):
    """Build a stable starting point for the rational curve fit."""

    mean_density = float(np.mean(density)) if density.size else 1.0
    guesses = []

    if grid.size >= 3:
        weights = np.sqrt(np.asarray(counts, dtype=float) + 1.0)
        try:
            p2, p1, p0 = np.polyfit(grid, density, deg=2, w=weights)
            guesses.append(np.array([float(p2), float(p1), max(float(p0), _RATIONAL_DENSITY_EPS), 0.0], dtype=float))
        except Exception:
            pass

    guesses.append(np.array([0.0, 0.0, max(mean_density, _RATIONAL_DENSITY_EPS), 0.0], dtype=float))
    if grid.size >= 2:
        slope = float(density[-1] - density[0]) / max(float(grid[-1] - grid[0]), _RATIONAL_DENSITY_EPS)
        guesses.append(np.array([0.0, slope, max(float(density[0]), _RATIONAL_DENSITY_EPS), 0.0], dtype=float))
        guesses.append(np.array([max(float(density.max() - density.min()), 0.0), 0.0, max(mean_density, _RATIONAL_DENSITY_EPS), 0.0], dtype=float))

    # Remove duplicated guesses while preserving order.
    unique_guesses = []
    for guess in guesses:
        if not any(np.allclose(guess, seen) for seen in unique_guesses):
            unique_guesses.append(guess)
    return unique_guesses


def _fit_rational_parameters_linear_search(grid, density, counts):
    """Fit the rational surrogate with a pure NumPy grid search over ``d``.

    For each candidate ``d`` we solve the remaining coefficients with weighted
    least squares on the linearized form of the model. This keeps the strategy
    usable when SciPy is not available in the execution environment.
    """

    grid = np.asarray(grid, dtype=float).ravel()
    density = np.asarray(density, dtype=float).ravel()
    counts = np.asarray(counts, dtype=float).ravel()

    if grid.size == 0 or grid.size != density.size or grid.size != counts.size:
        return None

    design = np.column_stack([grid**2, grid, np.ones_like(grid)])
    weights = np.sqrt(counts + 1.0)
    weighted_design = design * weights[:, None]

    best_params = None
    best_score = np.inf

    def _evaluate_candidates(d_candidates):
        nonlocal best_params, best_score
        for d in np.asarray(d_candidates, dtype=float).ravel():
            denominator = d * grid + 1.0
            if not np.all(np.isfinite(denominator)) or np.min(denominator) <= 1e-8:
                continue

            rhs = density * denominator
            weighted_rhs = rhs * weights

            try:
                abc, *_ = np.linalg.lstsq(weighted_design, weighted_rhs, rcond=None)
            except Exception:
                continue

            a, b, c = map(float, abc)
            if not np.all(np.isfinite([a, b, c])) or c < 0.0:
                continue

            fitted = _rational_quadratic_linear(grid, a, b, c, d)
            if np.any(~np.isfinite(fitted)) or np.min(fitted) < -_RATIONAL_DENSITY_EPS:
                continue

            score = float(np.average((fitted - density) ** 2, weights=weights))
            if score < best_score:
                best_score = score
                best_params = np.array([a, b, c, float(d)], dtype=float)

    coarse_candidates = np.linspace(-0.9, 8.0, 180)
    _evaluate_candidates(coarse_candidates)

    if best_params is not None:
        d0 = float(best_params[3])
        local_low = max(-0.9, d0 - 0.4)
        local_high = min(8.0, d0 + 0.4)
        if local_high > local_low:
            _evaluate_candidates(np.linspace(local_low, local_high, 120))

    return best_params


def _solve_rational_critical_points(a, b, c, d, *, tolerance=1e-12):
    """Solve the critical-point equation of the fitted rational function."""

    coef2 = float(a * d)
    coef1 = float(2.0 * a)
    coef0 = float(b - c * d)

    if abs(coef2) > tolerance:
        raw_roots = np.roots([coef2, coef1, coef0])
        roots = []
        for root in np.asarray(raw_roots).ravel():
            root = complex(root)
            if abs(root.imag) <= 1e-8 and np.isfinite(root.real):
                roots.append(float(root.real))
        return np.unique(np.asarray(roots, dtype=float)) if roots else np.array([], dtype=float)

    if abs(coef1) > tolerance:
        root = -coef0 / coef1
        if np.isfinite(root):
            return np.array([float(root)], dtype=float)

    return np.array([], dtype=float)


def _is_local_minimum(f, threshold, *, eps=_RATIONAL_LOCAL_EPS):
    """Check whether ``threshold`` is a local minimum of ``f`` by probing nearby."""

    probe = np.array([threshold - eps, threshold, threshold + eps], dtype=float)
    probe = np.clip(probe, 0.0, 1.0)
    values = np.asarray(f(probe), dtype=float).ravel()
    if values.size != 3 or np.any(~np.isfinite(values)):
        return False, values
    return bool(values[1] <= values[0] and values[1] <= values[2]), values


def _empty_threshold_report(*, requested, used, threshold, fallback_used=False, fallback_reason=None, overlap=None):
    """Build the common threshold diagnostics dictionary."""

    return {
        "threshold_strategy_requested": requested,
        "threshold_strategy_used": used,
        "threshold": float(threshold),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "separability_score": None,
        "rational_a": None,
        "rational_b": None,
        "rational_c": None,
        "rational_d": None,
        "fit_method": None,
        "fit_error": None,
        "quantile": None,
        "critical_points": [],
        "valid_minima": [],
        "grid_size": None,
        "n_bins": None,
        "margin": None,
        "overlap": overlap,
    }


class NoiseScoreFilter(BaseSampler):
    """Filter samples using a threshold on their noise scores.

    Parameters
    ----------
    sampling_strategy : str or dict, default="auto"
        Passed to :class:`imblearn.base.BaseSampler`.
    noise_filter : object or None, default=None
        Existing filter exposing a ``noise_score_`` attribute. When
        ``fit_filter=True`` and the filter is not already fitted, it is trained
        during :meth:`fit` to obtain those scores.
    noise_scores : array-like or None, default=None
        Precomputed noise scores. Values are clipped to ``[0, 1]`` during fit.
    fit_filter : bool, default=False
        Whether to fit ``noise_filter`` inside :meth:`fit` when it does not yet
        expose ``noise_score_``.
    threshold : {"mean", "quantile", "adaptative", "rational_valley", "adaptative_quadratic"} or float, default="mean"
        Thresholding rule. ``"mean"`` uses the average noise score.
        ``"quantile"`` uses the requested quantile of the score distribution.
        ``"adaptative"`` fits two beta-shaped score populations and uses their
        intersection as the threshold. ``"rational_valley"`` fits a rational
        surrogate to a density estimate and only accepts internal minima that
        are stable enough. Numeric values are used directly.
    quantile : float or None, default=None
        Quantile used when ``threshold="quantile"``.

    Notes
    -----
    The adaptive branch stores ``overlap_`` as the area shared by the two fitted
    densities. Lower values mean a cleaner separation between the low-score and
    high-score regions.
    """

    _sampling_type = "clean-sampling"

    def __init__(
        self,
        sampling_strategy="auto",
        noise_filter=None,
        noise_scores=None,
        threshold="mean",
        quantile=None,
        fit_filter=False,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.noise_filter = noise_filter
        self.noise_scores = noise_scores
        self.threshold = threshold
        self.threshold_report_ = None
        self.quantile = quantile
        self.fit_filter = fit_filter

    def _get_threshold(self):
        if self.threshold == "mean":
            self.overlap_ = None
            threshold = float(np.mean(self.noise_score_))
            self.threshold_report_ = _empty_threshold_report(requested=self.threshold, used="mean", threshold=threshold)
            return threshold

        if self.threshold == "quantile":
            if self.quantile is None:
                raise ValueError("quantile must be provided when threshold='quantile'.")
            quantile = float(self.quantile)
            if not np.isfinite(quantile):
                raise ValueError("quantile must be finite.")
            if not (0.0 <= quantile <= 1.0):
                raise ValueError("quantile must be in [0, 1].")
            self.overlap_ = None
            threshold = float(np.quantile(self.noise_score_, quantile))
            self.threshold_report_ = _empty_threshold_report(requested=self.threshold, used="quantile", threshold=threshold)
            self.threshold_report_["quantile"] = quantile
            return threshold

        if self.threshold == "adaptative":
            return self._get_adaptative_threshold()

        if self.threshold in {"rational_valley", "adaptative_quadratic"}:
            return self._get_rational_valley_threshold()

        if isinstance(self.threshold, (int, float, np.integer, np.floating)):
            threshold = float(self.threshold)
            if not np.isfinite(threshold):
                raise ValueError("threshold must be finite.")
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("threshold must be in [0, 1].")
            self.overlap_ = None
            self.threshold_report_ = _empty_threshold_report(requested=self.threshold, used="numeric", threshold=threshold)
            return threshold

        raise ValueError("threshold must be 'mean', 'adaptative', 'rational_valley', 'adaptative_quadratic', or a numeric value.")

    def _get_adaptative_threshold(self):
        """Estimate the threshold by intersecting two beta-shaped densities."""

        scores = np.asarray(self.noise_score_, dtype=float)
        if scores.size == 0:
            raise ValueError("noise_scores cannot be empty.")

        if np.any(~np.isfinite(scores)):
            raise ValueError("noise_scores must be finite.")

        # A nearly constant score vector does not provide enough structure for a
        # meaningful two-component fit, so we fall back to the empirical mean.
        if float(np.ptp(scores)) <= _BETA_EPS:
            self.overlap_ = 1.0
            threshold = float(np.clip(np.mean(scores), 0.0, 1.0))
            self.threshold_report_ = _empty_threshold_report(requested=self.threshold, used="adaptative", threshold=threshold, fallback_used=True, fallback_reason="constant_scores", overlap=self.overlap_)
            self.threshold_report_["grid_size"] = int(_ADAPTATIVE_GRID_SIZE)
            return threshold

        # Start from the median because it is less sensitive to extreme scores
        # than the mean and gives a more stable first split.
        threshold = float(np.median(scores))
        threshold = float(np.clip(threshold, 0.0, 1.0))

        grid = np.linspace(_BETA_EPS, 1.0 - _BETA_EPS, _ADAPTATIVE_GRID_SIZE)
        pdf_left = None
        pdf_right = None

        for _ in range(_ADAPTATIVE_MAX_ITER):
            left_mask = scores <= threshold
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                # If the current threshold sends everything to one side, split
                # the ordered scores in half and resume from there.
                order = np.argsort(scores, kind="mergesort")
                cut = max(1, min(scores.size - 1, scores.size // 2))
                left_mask = np.zeros(scores.size, dtype=bool)
                left_mask[order[:cut]] = True
                right_mask = ~left_mask

            left_scores = scores[left_mask]
            right_scores = scores[right_mask]
            alpha_left, beta_left = _fit_beta_moments(left_scores)
            alpha_right, beta_right = _fit_beta_moments(right_scores)
            mean_left = alpha_left / (alpha_left + beta_left)
            mean_right = alpha_right / (alpha_right + beta_right)

            # Evaluate the two fitted densities on the same open grid.
            log_pdf_left = _beta_log_pdf(grid, alpha_left, beta_left)
            log_pdf_right = _beta_log_pdf(grid, alpha_right, beta_right)

            weight_left = left_scores.size / scores.size
            weight_right = right_scores.size / scores.size

            log_weighted_left = np.log(weight_left + 1e-12) + log_pdf_left
            log_weighted_right = np.log(weight_right + 1e-12) + log_pdf_right

            pdf_left = np.exp(np.clip(log_weighted_left, -745.0, 700.0))
            pdf_right = np.exp(np.clip(log_weighted_right, -745.0, 700.0))

            new_threshold = _select_threshold_candidate(
                grid,
                log_weighted_left - log_weighted_right,
                mean_left,
                mean_right,
            )

            if abs(new_threshold - threshold) <= _ADAPTATIVE_TOL:
                threshold = new_threshold
                break
            threshold = new_threshold

        # The overlap is the shared area under both fitted densities.
        self.overlap_ = float(np.clip(_integrate_trapezoid(np.minimum(pdf_left, pdf_right), grid), 0.0, 1.0))
        threshold = float(np.clip(threshold, 0.0, 1.0))
        self.threshold_report_ = _empty_threshold_report(requested=self.threshold, used="adaptative", threshold=threshold, overlap=self.overlap_)
        self.threshold_report_["grid_size"] = int(_ADAPTATIVE_GRID_SIZE)
        return threshold

    def _get_rational_valley_threshold(self):
        """Estimate the threshold by fitting a rational surrogate to density.

        The fitted model is only accepted when it exposes a genuine internal
        minimum in the valid range and that minimum looks like a real valley
        rather than a flat or monotone shape.
        """

        scores = np.asarray(self.noise_score_, dtype=float).ravel()
        if scores.size == 0:
            raise ValueError("noise_scores cannot be empty.")
        if np.any(~np.isfinite(scores)):
            raise ValueError("noise_scores must be finite.")

        scores = np.clip(scores, 0.0, 1.0)
        fallback_threshold = float(np.clip(np.mean(scores), 0.0, 1.0))
        threshold_report = _empty_threshold_report(requested=self.threshold, used="rational_valley", threshold=fallback_threshold, fallback_used=True)
        threshold_report["margin"] = float(_RATIONAL_MARGIN)
        threshold_report["grid_size"] = int(_RATIONAL_GRID_SIZE)

        if scores.size < 5:
            threshold_report["fallback_reason"] = "density_estimation_failed"
            threshold_report["threshold_strategy_used"] = "rational_valley_fallback"
            self.threshold_report_ = threshold_report
            return fallback_threshold

        if float(np.ptp(scores)) <= _RATIONAL_MIN_SPREAD:
            threshold_report["fallback_reason"] = "constant_scores"
            threshold_report["threshold_strategy_used"] = "rational_valley_fallback"
            self.threshold_report_ = threshold_report
            return fallback_threshold

        try:
            grid, density, counts, n_bins = _estimate_histogram_density(scores)
        except Exception:
            threshold_report["fallback_reason"] = "density_estimation_failed"
            threshold_report["threshold_strategy_used"] = "rational_valley_fallback"
            self.threshold_report_ = threshold_report
            return fallback_threshold

        threshold_report["n_bins"] = int(n_bins)

        lower_bounds = [-np.inf, -np.inf, 0.0, -0.95]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        sigma = 1.0 / np.sqrt(np.asarray(counts, dtype=float) + 1.0)
        params = None
        last_error = None
        fit_method = None

        if curve_fit is not None:
            for initial in _initial_rational_parameters(grid, density, counts):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", OptimizeWarning)
                        params, _ = curve_fit(
                            _rational_quadratic_linear,
                            grid,
                            density,
                            p0=initial,
                            bounds=(lower_bounds, upper_bounds),
                            sigma=sigma,
                            maxfev=20000,
                        )
                    if params is not None:
                        fit_method = "curve_fit"
                        break
                except Exception as exc:
                    last_error = exc
                    params = None

        if params is not None:
            params = np.asarray(params, dtype=float).ravel()
            if params.size != 4 or np.any(~np.isfinite(params)):
                params = None
                fit_method = None

        if params is None:
            params = _fit_rational_parameters_linear_search(grid, density, counts)
            if params is not None:
                fit_method = "linear_grid_search"

        if params is None:
            threshold_report["fallback_reason"] = "fit_failed"
            threshold_report["threshold_strategy_used"] = "rational_valley_fallback"
            threshold_report["fit_error"] = repr(last_error) if last_error is not None else "linear_grid_search_failed"
            self.threshold_report_ = threshold_report
            return fallback_threshold

        threshold_report["fit_method"] = fit_method

        a, b, c, d = map(float, params)
        threshold_report["rational_a"] = a
        threshold_report["rational_b"] = b
        threshold_report["rational_c"] = c
        threshold_report["rational_d"] = d

        eval_grid = np.linspace(0.0, 1.0, _RATIONAL_GRID_SIZE)
        fitted_eval = _rational_quadratic_linear(eval_grid, a, b, c, d)
        if np.any(~np.isfinite(fitted_eval)):
            threshold_report["fallback_reason"] = "invalid_denominator"
            threshold_report["threshold_strategy_used"] = "rational_valley_fallback"
            self.threshold_report_ = threshold_report
            return fallback_threshold

        critical_points = _solve_rational_critical_points(a, b, c, d)
        critical_points = critical_points[np.isfinite(critical_points)]
        threshold_report["critical_points"] = [float(point) for point in critical_points.tolist()]

        in_bounds_candidates = []
        valid_minima = []
        for candidate in critical_points:
            if not (_RATIONAL_MARGIN <= float(candidate) <= 1.0 - _RATIONAL_MARGIN):
                continue
            in_bounds_candidates.append(float(candidate))

            candidate_value = float(_rational_quadratic_linear(np.array([candidate], dtype=float), a, b, c, d)[0])
            if not np.isfinite(candidate_value) or candidate_value < -_RATIONAL_DENSITY_EPS:
                continue
            candidate_value = float(max(candidate_value, 0.0))

            is_minimum, _ = _is_local_minimum(lambda x: _rational_quadratic_linear(x, a, b, c, d), float(candidate), eps=_RATIONAL_LOCAL_EPS)
            if is_minimum:
                valid_minima.append({"threshold": float(candidate), "density": candidate_value})

        threshold_report["valid_minima"] = valid_minima

        if not valid_minima:
            if critical_points.size == 0:
                threshold_report["fallback_reason"] = "no_internal_critical_point"
            elif not in_bounds_candidates:
                threshold_report["fallback_reason"] = "threshold_out_of_bounds"
            else:
                threshold_report["fallback_reason"] = "no_internal_minimum"
            threshold_report["threshold_strategy_used"] = "rational_valley_fallback"
            self.threshold_report_ = threshold_report
            return fallback_threshold

        best_minimum = min(valid_minima, key=lambda item: item["density"])
        threshold = float(best_minimum["threshold"])

        separability = 0.0
        left_mask = eval_grid < threshold
        right_mask = eval_grid > threshold
        if np.any(left_mask) and np.any(right_mask):
            fitted_non_negative = np.maximum(fitted_eval, 0.0)
            left_peak = float(np.max(fitted_non_negative[left_mask]))
            right_peak = float(np.max(fitted_non_negative[right_mask]))
            valley_density = float(max(_rational_quadratic_linear(np.array([threshold], dtype=float), a, b, c, d)[0], 0.0))
            denominator = min(left_peak, right_peak)
            if denominator > 0.0:
                separability = float(np.clip(1.0 - valley_density / denominator, 0.0, 1.0))

        threshold_report["separability_score"] = separability

        if separability < _RATIONAL_SEPARABILITY_MIN:
            threshold_report["fallback_reason"] = "low_separability"
            threshold_report["threshold_strategy_used"] = "rational_valley_fallback"
            self.threshold_report_ = threshold_report
            return fallback_threshold

        threshold = float(np.clip(threshold, 0.0, 1.0))
        threshold_report["threshold"] = threshold
        threshold_report["fallback_used"] = False
        threshold_report["fallback_reason"] = None
        threshold_report["threshold_strategy_used"] = "rational_valley"
        self.threshold_report_ = threshold_report
        return threshold

    def fit(self, X, y):
        """Cache the noise scores, optionally fit the nested filter and compute the threshold."""

        y = _as_1d_array(y, name="y")
        X = np.asarray(X)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.overlap_ = None
        self.threshold_report_ = None
        self.noise_filter_ = None

        if self.noise_filter is not None:
            noise_score = getattr(self.noise_filter, "noise_score_", None)
            if noise_score is None:
                if not self.fit_filter:
                    raise AttributeError("noise_filter must have a noise_score_ attribute or set fit_filter=True.")
                if not hasattr(self.noise_filter, "fit"):
                    raise AttributeError("noise_filter must have a fit method when fit_filter=True.")
                fitted_noise_filter = self.noise_filter.fit(X, y)
                self.noise_filter_ = fitted_noise_filter if fitted_noise_filter is not None else self.noise_filter
                noise_score = getattr(self.noise_filter_, "noise_score_", None)
            else:
                self.noise_filter_ = self.noise_filter

            if noise_score is None:
                raise AttributeError("noise_filter must expose a noise_score_ attribute after fitting.")
            self.noise_score_ = np.asarray(_as_1d_array(noise_score, name="noise_filter.noise_score_"), dtype=float)
        elif self.noise_scores is not None:
            self.noise_score_ = np.asarray(_as_1d_array(self.noise_scores, name="noise_scores"), dtype=float)
        else:
            raise ValueError("Provide either noise_filter or noise_scores.")

        if self.noise_score_.shape[0] != len(y):
            raise ValueError("noise_scores and y must have the same number of samples.")
        if self.noise_score_.size == 0:
            raise ValueError("noise_scores cannot be empty.")
        if np.any(~np.isfinite(self.noise_score_)):
            raise ValueError("noise_scores must be finite.")

        # The filter assumes normalized scores, so we keep the public API permissive
        # and clip minor out-of-range values instead of failing on them.
        self.noise_score_ = np.clip(self.noise_score_, 0.0, 1.0)

        self.threshold_ = self._get_threshold()
        self.keep_mask_ = self.noise_score_ <= self.threshold_
        return self

    def _fit_resample(self, X, y):
        X = np.asarray(X)
        y = _as_1d_array(y, name="y")
        return X[self.keep_mask_], y[self.keep_mask_]

    def fit_resample(self, X, y):
        """Fit the filter and return the cleaned sample set."""

        self.fit(X, y)
        return self._fit_resample(X, y)
