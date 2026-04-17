# numba_fit_template.py
import numpy as np
from numba import njit
from typing import Tuple, Dict, Any
import warnings

# ---------- Numba-compiled numeric kernels ----------

# Add multiband offset computation and fitting

@njit
def construct_gamma_numba(t, band_of_obs, phi, omega, templates):
    """
    Vectorized periodic linear interpolation of templates for each observation.
    templates: shape (n_bands, M)
    t: times (centered), band_of_obs: int indices 0..n_bands-1

    NOTE: changed mapping to use idxf = phase * M and wrap indices modulo M so
    interpolation is periodic and consistent with the callable-based R/Python templates.
    """
    n_obs = t.shape[0]
    n_bands, M = templates.shape
    out = np.empty(n_obs, dtype=np.float64)
    for i in range(n_obs):
        phase = (t[i] * omega + phi) % 1.0
        # fractional index in [0, M)
        idxf = phase * M
        i0 = int(np.floor(idxf)) % M
        frac = idxf - np.floor(idxf)
        i1 = (i0 + 1) % M
        b = band_of_obs[i]
        y0 = templates[b, i0]
        y1 = templates[b, i1]
        out[i] = y0 * (1.0 - frac) + y1 * frac
    return out

@njit
def compute_beta_multiband(m, dust, gammaf, band_of_obs, n_bands, weights, use_errors_int):
    """
    Solve weighted least squares for mu, d, a, and band offsets using
    X = [1, dust, gammaf, offset_1, offset_2, ..., offset_(n_bands-1)]
    (n_bands - 1 offsets; one band is reference with offset = 0)
    
    Returns array shape (3 + n_bands - 1,) = [mu, d, a, offset_1, ..., offset_(n_bands-1)]
    """
    n = m.shape[0]
    n_params = 3 + n_bands - 1
    
    # Build normal matrix and RHS
    B = np.zeros((n_params, n_params), dtype=np.float64)
    dvec = np.zeros(n_params, dtype=np.float64)
    
    if use_errors_int != 0:
        for i in range(n):
            w = weights[i]
            # Build design matrix row: [1, dust, gammaf, offset_indicators]
            x = np.zeros(n_params)
            x[0] = 1.0
            x[1] = dust[i]
            x[2] = gammaf[i]
            # offset indicators (one-hot for each band except the last)
            b = band_of_obs[i]
            if b < n_bands - 1:
                x[3 + b] = 1.0
            
            # X'WX and X'Wy
            for j in range(n_params):
                dvec[j] += x[j] * m[i] * w
                for k in range(n_params):
                    B[j, k] += x[j] * x[k] * w
    else:
        for i in range(n):
            x = np.zeros(n_params)
            x[0] = 1.0
            x[1] = dust[i]
            x[2] = gammaf[i]
            b = band_of_obs[i]
            if b < n_bands - 1:
                x[3 + b] = 1.0
            
            for j in range(n_params):
                dvec[j] += x[j] * m[i]
                for k in range(n_params):
                    B[j, k] += x[j] * x[k]
    
    # Regularize and solve
    B += 1e-10 * np.eye(n_params)
    z = np.linalg.solve(B, dvec)
    return z

@njit
def compute_beta_one_multiband(m, gammaf, band_of_obs, n_bands, weights, use_errors_int):
    """
    Solve for mu, a, and band offsets (no dust).
    X = [1, gammaf, offset_1, ..., offset_(n_bands-1)]
    Returns shape (2 + n_bands - 1,) = [mu, a, offset_1, ..., offset_(n_bands-1)]
    """
    n = m.shape[0]
    n_params = 2 + n_bands - 1
    
    B = np.zeros((n_params, n_params), dtype=np.float64)
    dvec = np.zeros(n_params, dtype=np.float64)
    
    if use_errors_int != 0:
        for i in range(n):
            w = weights[i]
            x = np.zeros(n_params)
            x[0] = 1.0
            x[1] = gammaf[i]
            b = band_of_obs[i]
            if b < n_bands - 1:
                x[2 + b] = 1.0
            
            for j in range(n_params):
                dvec[j] += x[j] * m[i] * w
                for k in range(n_params):
                    B[j, k] += x[j] * x[k] * w
    else:
        for i in range(n):
            x = np.zeros(n_params)
            x[0] = 1.0
            x[1] = gammaf[i]
            b = band_of_obs[i]
            if b < n_bands - 1:
                x[2 + b] = 1.0
            
            for j in range(n_params):
                dvec[j] += x[j] * m[i]
                for k in range(n_params):
                    B[j, k] += x[j] * x[k]
    
    B += 1e-10 * np.eye(n_params)
    z = np.linalg.solve(B, dvec)
    return z

@njit
def compute_beta_numba(m, dust, gammaf, weights, use_errors_int):
    """
    Solve weighted least squares for mu,d,a using X = [1, dust, gammaf].
    use_errors_int == 1 => use weights, else unweighted.
    Returns array shape (3,)
    """
    n = m.shape[0]
    B00 = B01 = B02 = B11 = B12 = B22 = 0.0
    d0 = d1 = d2 = 0.0
    if use_errors_int != 0:
        for i in range(n):
            w = weights[i]
            x1 = dust[i]
            x2 = gammaf[i]
            B00 += 1.0 * 1.0 * w
            B01 += 1.0 * x1 * w
            B02 += 1.0 * x2 * w
            B11 += x1 * x1 * w
            B12 += x1 * x2 * w
            B22 += x2 * x2 * w
            d0 += 1.0 * m[i] * w
            d1 += x1 * m[i] * w
            d2 += x2 * m[i] * w
    else:
        for i in range(n):
            x1 = dust[i]
            x2 = gammaf[i]
            B00 += 1.0
            B01 += x1
            B02 += x2
            B11 += x1 * x1
            B12 += x1 * x2
            B22 += x2 * x2
            d0 += m[i]
            d1 += x1 * m[i]
            d2 += x2 * m[i]
    # symmetric fill
    B10 = B01
    B20 = B02
    B21 = B12
    # Build matrix and vector
    B = np.empty((3,3), dtype=np.float64)
    B[0,0] = B00; B[0,1] = B01; B[0,2] = B02
    B[1,0] = B10; B[1,1] = B11; B[1,2] = B12
    B[2,0] = B20; B[2,1] = B21; B[2,2] = B22
    dvec = np.array((d0, d1, d2), dtype=np.float64)
    B += 1e-10 * np.eye(B.shape[0])
    z = np.linalg.solve(B, dvec)
    return z

@njit
def compute_beta_one_numba(m, gammaf, weights, use_errors_int):
    """
    Solve for mu,a (no dust). X = [1, gammaf]
    Returns shape (2,)
    """
    n = m.shape[0]
    B00 = B01 = B11 = 0.0
    d0 = d1 = 0.0
    if use_errors_int != 0:
        for i in range(n):
            w = weights[i]
            x1 = gammaf[i]
            B00 += 1.0 * 1.0 * w
            B01 += 1.0 * x1 * w
            B11 += x1 * x1 * w
            d0 += 1.0 * m[i] * w
            d1 += x1 * m[i] * w
    else:
        for i in range(n):
            x1 = gammaf[i]
            B00 += 1.0
            B01 += x1
            B11 += x1 * x1
            d0 += m[i]
            d1 += x1 * m[i]
    B10 = B01
    B = np.empty((2,2), dtype=np.float64)
    B[0,0] = B00; B[0,1] = B01; B[1,0] = B10; B[1,1] = B11
    dvec = np.array((d0, d1), dtype=np.float64)
    B += 1e-10 * np.eye(B.shape[0])
    z = np.linalg.solve(B, dvec)
    return z

@njit
def newton_update_numba(phi, omega, m, t, dust, weights, band_of_obs, templates, templated):
    """
    One Newton step: compute mu,d,a and (if a>0) update phi using templatedFuncs.
    Returns [mu,d,a,phi_new]
    """
    gammaf = construct_gamma_numba(t, band_of_obs, phi, omega, templates)
    dust_nonzero = 0
    for i in range(dust.shape[0]):
        if dust[i] != 0.0:
            dust_nonzero = 1
            break
    if dust_nonzero != 0:
        est = compute_beta_numba(m, dust, gammaf, weights, 1)
        mu = est[0]; d = est[1]; a = est[2]
    else:
        est = compute_beta_one_numba(m, gammaf, weights, 1)
        mu = est[0]; a = est[1]; d = 0.0

    if a > 0.0:
        gammafd = construct_gamma_numba(t, band_of_obs, phi, omega, templated)
        n = m.shape[0]
        mp = np.empty(n, dtype=np.float64)
        for i in range(n):
            mp[i] = m[i] - mu - d * dust[i]
        delv = 0.0
        h = 0.0
        for i in range(n):
            w = weights[i]
            delv += gammafd[i] * (mp[i] - a * gammaf[i]) * w
            h += a * gammafd[i] * gammafd[i] * w
        if h == 0.0:
            phi_new = np.random.rand()
        else:
            phi_new = (phi + delv / h) % 1.0
    else:
        a = 0.0
        phi_new = np.random.rand()
    out = np.empty(4, dtype=np.float64)
    out[0] = mu; out[1] = d; out[2] = a; out[3] = phi_new
    return out


@njit
def newton_update_multiband(phi, omega, m, t, dust, weights, band_of_obs, 
                                     templates, templated, n_bands, a_max = 5.0):
    """
    One Newton step with band offsets.
    Returns [mu, d, a, phi_new, offset_1, ..., offset_(n_bands-1)]
    """
    gammaf = construct_gamma_numba(t, band_of_obs, phi, omega, templates)
    dust_nonzero = 0
    for i in range(dust.shape[0]):
        if dust[i] != 0.0:
            dust_nonzero = 1
            break
    
    if dust_nonzero != 0:
        est = compute_beta_multiband(m, dust, gammaf, band_of_obs, n_bands, weights, 1)
        mu = est[0]
        d = est[1]
        a = est[2]
        offsets = est[3:]
    else:
        est = compute_beta_one_multiband(m, gammaf, band_of_obs, n_bands, weights, 1)
        mu = est[0]
        a = est[1]
        d = 0.0
        offsets = est[2:]

    if a > a_max: # add amp bounds 
        a = a_max
    
    if a > 0.0: # and a < a_max:
        gammafd = construct_gamma_numba(t, band_of_obs, phi, omega, templated)
        n = m.shape[0]
        mp = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            offset_i = 0.0
            b = band_of_obs[i]
            if b < n_bands - 1:
                offset_i = offsets[b]
            mp[i] = m[i] - mu - d * dust[i] - offset_i
        
        delv = 0.0
        h = 0.0
        for i in range(n):
            w = weights[i]
            delv += gammafd[i] * (mp[i] - a * gammaf[i]) * w
            h += a * gammafd[i] * gammafd[i] * w
        
        if h == 0.0:
            phi_new = np.random.rand()
        else:
            phi_new = (phi + delv / h) % 1.0
    else:
        a = 0.0
        phi_new = np.random.rand()
    
    n_params = 4 + n_bands - 1
    out = np.empty(n_params, dtype=np.float64)
    out[0] = mu
    out[1] = d
    out[2] = a
    out[3] = phi_new
    for j in range(n_bands - 1):
        out[4 + j] = offsets[j]
    return out

@njit
def chi2_for_fixed_phi_with_offset(phi, omega, m_temp, t, dust_obs, weights,
                                    band_of_obs, templates, n_bands):
    """
    Compute chi2 with band offsets for a single (omega, phi).
    """
    gammaf = construct_gamma_numba(t, band_of_obs, phi, omega, templates)
    
    n = m_temp.shape[0]
    dust_nonzero = 0
    for i in range(n):
        if dust_obs[i] != 0.0:
            dust_nonzero = 1
            break
    
    if dust_nonzero != 0:
        est = compute_beta_multiband(m_temp, dust_obs, gammaf, band_of_obs, n_bands, weights, 1)
        mu = est[0]
        d = est[1]
        a = est[2]
        offsets = est[3:]
    else:
        est = compute_beta_one_multiband(m_temp, gammaf, band_of_obs, n_bands, weights, 1)
        mu = est[0]
        a = est[1]
        d = 0.0
        offsets = est[2:]
    
    if a < 0.0:
        a = 0.0
    
    chi2 = 0.0
    for i in range(n):
        offset_i = 0.0
        b = band_of_obs[i]
        if b < n_bands - 1:
            offset_i = offsets[b]
        r = m_temp[i] - mu - d * dust_obs[i] - a * gammaf[i] - offset_i
        chi2 += weights[i] * r * r
    
    return chi2

# ---------- helpers to prepare arrays (Python side) ----------

def prepare_arrays_for_numba(tem_py: Dict[str, Any], lc_py, cols: list = ['time', 'band', 'mag', 'error'],
                             use_errors: bool = True, use_dust: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Convert template dict and pandas DataFrame lc_py into numpy arrays for numba functions.

    Returns:
      t (centered), m, error, weights, dust_obs, band_of_obs (int), templates, templated, mean_time

    This now:
      - reorders/subsets template dict to match the bands present in lc (same behavior as R CheckTemLC),
      - adds tem['model_error'] in quadrature to per-observation errors when use_errors is True,
      - disables dust-fitting when only a single band is present (like R's CheckNumberBands) and warns.
    """
    # Sort by band to match R AugmentData behavior
    lc_sorted = lc_py.sort_values(cols[1], kind="stable").reset_index(drop=True)

    # If lc has only one unique band, R switches off fitting dust.
    lc_unique_bands = list(lc_sorted[cols[1]].unique())
    if use_dust and len(lc_unique_bands) == 1:
        warnings.warn("light curve has only 1 band: disabling dust fit (E[B-V]=0) to match R behavior")
        use_dust = False

    # Reorder / subset template to match the order of bands in the LC (CheckTemLC + AugmentData behavior)
    tem_reordered = reorder_template_for_lc(tem_py, lc_unique_bands)

    bands = tem_reordered["bands"]
    band_index_map = {b: i for i, b in enumerate(bands)}
    n_obs = lc_sorted.shape[0]
    band_of_obs = np.empty(n_obs, dtype=np.int64)
    for i, b in enumerate(lc_sorted[cols[1]].values):
        band_of_obs[i] = band_index_map[b]
    t = lc_sorted[cols[0]].values.astype(np.float64)
    mean_time = float(np.mean(t))
    t_center = t - mean_time
    m = lc_sorted[cols[2]].values.astype(np.float64)
    error = lc_sorted[cols[3]].values.astype(np.float64)

    # Incorporate model_error (per-band) in quadrature as R's AugmentData does when use_errors is True
    if use_errors:
        model_error_arr = np.empty(n_obs, dtype=np.float64)
        for i in range(n_obs):
            model_error_arr[i] = float(tem_reordered["model_error"][bands[band_of_obs[i]]])
        error = np.sqrt(error * error + model_error_arr * model_error_arr)
    # weights
    weights = 1.0 / (error ** 2)

    # dust per observation (zeroed out if we're not using dust)
    dust_obs = np.empty(n_obs, dtype=np.float64)
    if use_dust:
        for i in range(n_obs):
            dust_obs[i] = float(tem_reordered["dust"][bands[band_of_obs[i]]])
    else:
        dust_obs.fill(0.0)

    templates = np.asarray(tem_reordered["templates"], dtype=np.float64)
    templated = np.asarray(tem_reordered["templatesd"], dtype=np.float64)
    # Ensure shape: templates (n_bands, M)
    if templates.shape[0] != len(bands) and templates.shape[1] == len(bands):
        templates = templates.T
    if templated.shape[0] != len(bands) and templated.shape[1] == len(bands):
        templated = templated.T
    return t_center, m, error, weights, dust_obs, band_of_obs, templates, templated, mean_time

def reorder_coeffs_to_filter_order(coeffs, cov, lc_bands, filter_order):
    """
    Reorder fitted coefficients and covariance from LC band order back to canonical filter order.
    
    Parameters
    ----------
    coeffs : np.ndarray
        Fitted coefficients [mu, d, a, phi, offset_lc0, offset_lc1, ..., offset_lc(n-2)]
        where indices 0..n-2 correspond to lc_bands[0..n-2]
    cov : np.ndarray
        Covariance matrix in LC band order (shape: (3+n_bands-1, 3+n_bands-1))
    lc_bands : list
        Bands in the order they appeared during fitting (from LC unique/sorted)
    filter_order : list
        Canonical filter order (e.g., ['g', 'r', 'i', 'z'])
    
    Returns
    -------
    coeffs_reordered : np.ndarray
        Coefficients reordered to filter_order
    cov_reordered : np.ndarray
        Covariance matrix reordered to filter_order
    """
    # Verify all LC bands are in filter order
    missing = [b for b in lc_bands if b not in filter_order]
    if missing:
        raise ValueError(f"LC bands {missing} not found in filter_order {filter_order}")
    
    # Extract mu, d, a, phi (these don't change)
    mu, d, a, phi = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    lc_offsets = coeffs[4:]
    
    # Create mapping from lc_bands to filter_order indices
    # lc_bands[i] -> position in filter_order
    lc_to_filter_idx = {band: filter_order.index(band) for band in lc_bands}
    
    # Build reordered offset array for all filters
    # Last band (reference) has offset = 0
    n_filters = len(filter_order)
    offsets_reordered = np.zeros(n_filters - 1, dtype=np.float64)
    
    for lc_idx, band in enumerate(lc_bands):
        filter_idx = lc_to_filter_idx[band]
        
        # The last band in lc_bands has offset=0 (reference)
        if lc_idx < len(lc_bands) - 1:
            # This band has a fitted offset
            offsets_reordered[filter_idx] = lc_offsets[lc_idx]
        # else: band is reference in LC order, offset stays 0
    
    # Rebuild coeffs in filter order
    coeffs_reordered = np.concatenate([
        [mu, d, a, phi],
        offsets_reordered
    ])
    
    # ---- Reorder covariance matrix ----
    # Covariance structure in LC order:
    #   [[mu, mu], [mu, d], [mu, a], [mu, o0], [mu, o1], ...]
    #   [[d,  mu], [d,  d], [d,  a], [d,  o0], [d,  o1], ...]
    #   etc.
    # We only reorder the offset rows/cols (indices 3:)
    
    n_lc_bands = len(lc_bands)
    n_lc_params = 3 + n_lc_bands - 1
    
    # Build permutation for offsets
    # Old offset indices: [3, 4, 5, ..., 3+n_lc_bands-2]
    # New offset indices: should go to filter_order positions
    
    # Create full permutation array
    perm = [0, 1, 2]  # mu, d, a stay in place
    
    # Add offset permutation
    for lc_idx in range(n_lc_bands - 1):
        band = lc_bands[lc_idx]
        filter_idx = lc_to_filter_idx[band]
        perm.append(3 + filter_idx)
    
    # Apply permutation to covariance matrix
    perm = np.array(perm)
    cov_reordered = cov[np.ix_(perm, perm)]
    
    return coeffs_reordered, cov_reordered

# ---------- high-level compiled wrappers (use these) ----------

def FitTemplate_multiband(tem_py: Dict[str, Any], lc_py, omegas: np.ndarray, NN: int = 5,
                      cols: list = ['time', 'band', 'mag', 'error'], use_errors: bool = True, 
                      use_dust: bool = False, use_band_shift: bool = True, tol: float = 1e-6) -> np.ndarray:
    """
    Compute RSS for each omega using numba-compiled inner loops.
    
    Parameters
    ----------
    tem_py : dict
        Template dictionary
    lc_py : DataFrame
        Light curve data
    omegas : np.ndarray
        Angular frequencies to test
    NN : int
        Number of Newton steps per restart
    cols : list
        Column names [time, band, mag, error]
    use_errors : bool
        Whether to use per-observation errors in weighting
    use_dust : bool
        Whether to fit dust extinction (disabled if use_band_shift=True)
    use_band_shift : bool
        Whether to fit per-band magnitude offsets (disables dust fitting)
    tol : float
        tolerance to determine if phi is converged
    
    Returns
    -------
    rss : np.ndarray
        Residual sum of squares for each omega
    
    Notes
    -----
    If use_band_shift is True, use_dust is automatically set to False to avoid degeneracy.
    """
    
    # Force dust off if using band shifts
    if use_band_shift:
        if use_dust:
            warnings.warn("use_band_shift=True disables dust fitting to avoid degeneracy")
        use_dust = False
    
    # prepare arrays (now applies model error in quadrature and reorders template to match lc bands)
    t, m, error, weights, dust_obs, band_of_obs, templates, templated, mean_time = \
        prepare_arrays_for_numba(tem_py, lc_py, cols, use_errors=use_errors, use_dust=use_dust)
    
    if not use_errors:
        weights = np.ones_like(weights)
    
    omegas = np.asarray(omegas, dtype=np.float64)
    n_omegas = omegas.shape[0]
    n_obs = m.shape[0]
    
    # Get band information
    lc_bands = list(lc_py[cols[1]].sort_values(kind="stable").unique())
    n_bands = len(lc_bands)
    
    # compute betas outside numba via tem_reordered['abs_mag']
    periods = 1.0 / omegas
    tem_for_abs = reorder_template_for_lc(tem_py, lc_bands)
    betas = tem_for_abs["abs_mag"](periods, tem_for_abs)  # shape (n_omegas, n_bands)
    
    # allocate rss
    rss = np.empty(n_omegas, dtype=np.float64)
    
    # main loop over omegas
    for i in range(n_omegas):
        omega = omegas[i]
        betas_row = betas[i, :]  # length n_bands
        
        # subtract per-observation absolute mags
        m_temp = np.empty(n_obs, dtype=np.float64)
        for j in range(n_obs):
            m_temp[j] = m[j] - betas_row[band_of_obs[j]]
        
        # initialize coefficients
        phi = np.random.rand()
        mu = 0.0
        d = 0.0
        a = 0.0
        band_shifts = np.zeros(n_bands - 1, dtype=np.float64) if use_band_shift else np.empty(0, dtype=np.float64)
        
        if use_band_shift:
            # NN Newton steps with band shifts (warm start)
            iters = 0
            diff = np.inf
            # while iters < NN and diff > tol:
            for _ in range(NN):
                
                phi_old = phi
                out = newton_update_multiband(phi, omega, m_temp, t, dust_obs, weights, 
                                                       band_of_obs, templates, templated, n_bands)
                mu = out[0]
                d = out[1]
                a = out[2]
                phi = out[3]
                band_shifts = out[4:]
                iters += 1
                diff = np.abs(phi - phi_old)
        else:
            # NN Newton steps without band shifts
            for _ in range(NN):
                out = newton_update_numba(phi, omega, m_temp, t, dust_obs, weights, 
                                          band_of_obs, templates, templated)
                mu, d, a, phi = out[0], out[1], out[2], out[3]
        
        # final gamma and residuals
        gammaf = construct_gamma_numba(t, band_of_obs, phi, omega, templates)
        resid_sum = 0.0
        
        if use_band_shift:
            for j in range(n_obs):
                b = band_of_obs[j]
                offset_j = band_shifts[b] if b < n_bands - 1 else 0.0
                rj = m_temp[j] - mu - d * dust_obs[j] - a * gammaf[j] - offset_j
                resid_sum += weights[j] * rj * rj
        else:
            for j in range(n_obs):
                rj = m_temp[j] - mu - d * dust_obs[j] - a * gammaf[j]
                resid_sum += weights[j] * rj * rj
        
        rss[i] = resid_sum
    
    return rss
    
def ComputeCoeffs_multiband(tem_py: Dict[str, Any], lc_py, omega: float, NN: int = 20,
                                     cols: list = ['time', 'band', 'mag', 'error'], 
                                     use_errors: bool = True, use_dust: bool = False) -> np.ndarray:
    """
    For a single omega, compute [mu, d, a, phi, offset_1, ..., offset_(n_bands-1)].
    
    Parameters
    ----------
    use_dust : bool
        Disable dust fitting when using multiband offsets to avoid degeneracy.
    
    Returns
    -------
    coeffs : ndarray
        [mu, d, a, phi, offset_1, offset_2, ..., offset_(n_bands-1)]
        Last band offset is implicitly 0 (reference band).
    """
    t, m, error, weights, dust_obs, band_of_obs, templates, templated, mean_time = \
        prepare_arrays_for_numba(tem_py, lc_py, cols, use_errors=use_errors, use_dust=False)  # Force no dust internally
    
    if not use_errors:
        weights = np.ones_like(weights)
    
    # Get number of bands
    lc_bands = list(lc_py[cols[1]].sort_values(kind="stable").unique())
    n_bands = len(lc_bands)
    
    # Subtract absolute magnitude
    tem_for_abs = reorder_template_for_lc(tem_py, lc_bands)
    betas_row = tem_for_abs["abs_mag"]([1.0 / omega], tem_for_abs)[0]
    
    n_obs = m.shape[0]
    m_temp = np.empty(n_obs, dtype=np.float64)
    for j in range(n_obs):
        m_temp[j] = m[j] - betas_row[band_of_obs[j]]
    
    # Initialize and iterate
    phi = np.random.rand()
    J = 0
    a = 0.0
    band_shifts = np.zeros(n_bands - 1, dtype=np.float64)
    
    while a == 0.0 and J < 10:
        for _ in range(NN):
            out = newton_update_multiband(phi, omega, m_temp, t, dust_obs, weights, 
                                                   band_of_obs, templates, templated, n_bands)
            mu, d, a, phi = out[0], out[1], out[2], out[3]
            band_shifts = out[4:]
        J += 1
    
    # Shift phase back
    phi_shifted = (phi - omega * mean_time) % 1.0
    
    # Extract offsets
    n_params = 4 + n_bands - 1
    coeffs = np.empty(n_params, dtype=np.float64)
    coeffs[0] = mu
    coeffs[1] = d
    coeffs[2] = a
    coeffs[3] = phi_shifted
    for j in range(n_bands - 1):
        coeffs[4 + j] = band_shifts[j]
    
    return coeffs

def ComputeCoeffsAndCov_multiband(
    tem_py: Dict[str, Any], lc_py, omega: float, NN: int = 20,
    cols: list = ['time', 'band', 'mag', 'error'],
    use_errors: bool = True,
    use_dust: bool = False,
    filter_order: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coefficients and covariance matrix for multiband fitting with band offsets.
    
    Parameters
    ----------
    tem_py : dict
        Template dictionary
    lc_py : DataFrame
        Light curve data
    omega : float
        Angular frequency
    NN : int
        Number of Newton steps per restart
    cols : list
        Column names [time, band, mag, error]
    use_errors : bool
        Whether to use per-observation errors in weighting
    use_dust : bool
        Should be False (band shifts replace dust fitting)
    
    Returns
    -------
    coeffs : np.ndarray
        [mu, d, a, phi, offset_1, offset_2, ..., offset_(n_bands-1)]
    cov : np.ndarray
        Covariance matrix for all parameters (shape: (3+n_bands-1, 3+n_bands-1))
        covers [mu, d, a, offset_1, ..., offset_(n_bands-1)]
        Note: phi is not included in covariance (nonlinear parameter)
    """
    
    # ---- Run existing solver ----
    coeffs = ComputeCoeffs_multiband(
        tem_py, lc_py, omega,
        NN=NN, cols=cols,
        use_errors=use_errors,
        use_dust=False  # Force no dust
    )
    
    mu = coeffs[0]
    d = coeffs[1]
    a = coeffs[2]
    phi = coeffs[3]
    band_offsets = coeffs[4:]
    
    # ---- Reconstruct arrays ----
    t, m, error, weights, dust_obs, band_of_obs, templates, templated, _ = \
        prepare_arrays_for_numba(
            tem_py, lc_py, cols,
            use_errors=use_errors,
            use_dust=False
        )
    
    # subtract absolute magnitude
    lc_bands = list(lc_py[cols[1]].sort_values(kind="stable").unique())
    n_bands = len(lc_bands)
    tem_for_abs = reorder_template_for_lc(tem_py, lc_bands)
    betas = tem_for_abs["abs_mag"]([1.0/omega], tem_for_abs)[0]
    
    m_temp = np.empty_like(m)
    for i in range(len(m)):
        m_temp[i] = m[i] - betas[band_of_obs[i]]
    
    # ---- Build design matrix X with band offsets ----
    gammaf = construct_gamma_numba(t, band_of_obs, phi, omega, templates)
    
    n_obs = len(m)
    n_params = 3 + n_bands - 1  # mu, d, a, plus n_bands-1 offsets
    X = np.empty((n_obs, n_params), dtype=np.float64)
    
    # Columns: [1, dust, gamma, offset_0, offset_1, ..., offset_(n-2)]
    X[:, 0] = 1.0
    X[:, 1] = dust_obs
    X[:, 2] = gammaf
    
    # Band offset columns (one-hot encoding)
    for i in range(n_obs):
        b = band_of_obs[i]
        for j in range(n_bands - 1):
            if b == j:
                X[i, 3 + j] = 1.0
            else:
                X[i, 3 + j] = 0.0
    
    # ---- Weight matrix ----
    W = np.diag(weights)
    
    # ---- Covariance ----
    XT_W = X.T @ W
    B = XT_W @ X  # Xᵀ W X
    
    # Regularize before inversion
    B += 1e-10 * np.eye(n_params)
    B_inv = np.linalg.inv(B)
    
    # ---- χ² and scaling ----
    # Compute residuals accounting for band offsets
    resid = np.empty(n_obs, dtype=np.float64)
    for i in range(n_obs):
        b = band_of_obs[i]
        offset_i = band_offsets[b] if b < n_bands - 1 else 0.0
        resid[i] = m_temp[i] - mu - d * dust_obs[i] - a * gammaf[i] - offset_i

    # print(weights * resid**2)
    chi2 = np.sum(weights * resid**2)
    dof = n_obs - n_params
    chi2_red = chi2 / dof if dof > 0 else 1.0
#     print(chi2_red)
    # Covariance matrix scaled by reduced chi-squared
    cov = chi2_red * B_inv
    #Put output back in ugrizy order
    # lc_bands = list(lc_py[cols[1]].sort_values(kind="stable").unique())

    # coeffs_reordered, cov_reordered = reorder_coeffs_to_filter_order(
    #     coeffs, cov, lc_bands, filter_order)
    
    # return coeffs_reordered, cov_reordered
    
    return coeffs, cov

# -------------- Forward Facing Helper functions (require more imports) ------------------------
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
filter_colors ={'u': '#0c71ff', 'g': '#49be61', 'r': '#c61c00', 'i': '#ffc200', 'z': '#f341a2', 'y': '#5d0000'}

def template_fitting(tem, lc, print_outputs = False, fit_n = 20, coeff_n = 10, omega_n = 100.0, cols=['midpointMjdTai', 'band', 'psfMag', 'psfMagErr']):
    '''
    Returns a dictionary with coeffs (mu, d, a, phi), pests (top 3), cov (of linear params: mu (distance modulus), d (dust), a (amplitude)), sigma_P (local uncertainty of a given period), 
        like_P (global posterior likelihood uncertainty), the next best 3 periods, gen_lc (the generated light curve)
    '''
    # compute best coefficients
    omegas = np.arange(1.1, 5.0, 0.1/omega_n) #periods from [0.2, 0.9]
    rss = FitTemplate_multiband(tem, lc, omegas, NN=fit_n, use_errors=True, use_dust=False, use_band_shift=True, cols = cols)
    rss = np.array(rss)
    
    best_omega = omegas[np.argmin(rss)]
    best_pest = 1/best_omega
    
    coeffs, cov = ComputeCoeffsAndCov_multiband(tem, lc, float(best_omega), NN=coeff_n, use_errors=True, use_dust=False, cols = cols)

    # calculate error and posterior on period
    chi2_min = np.min(rss) # rss is chi2 bc it's already scaled by weights

    chi2_red = chi2_min / (len(lc.midpointMjdTai) - 4) # length - dof
        
    periods = 1/np.array(omegas)
    rms = np.sqrt(rss / (len(lc.midpointMjdTai) - 4)) # length - dof
    mask = rss <= chi2_min + 2.3
   
    sigma_P = 0.5 * np.abs(periods[mask][0] - periods[mask][-1])

    # now get posterior likelihood to figure out how likely this is to be the right answer
    peaks, props = find_peaks(-rss, prominence=10, width=0.1)
    next_best_idx = np.argpartition(props['prominences'], -3)[-3:]
    next_best = periods[peaks[next_best_idx]]
    next_best_chi = rss[peaks[next_best_idx]]
   
    L = np.exp(-0.5 * (rss)) #convert to posterior - rss.min()
    L /= np.trapezoid(L, periods) # probability density
    P_mean = np.trapezoid(periods * L, periods) # mean of the posterior
    P_var  = np.trapezoid((periods - P_mean)**2 * L, periods) #variance on the posterior
    like_P = np.sqrt(P_var) #std of posterior

    if print_outputs:
        # print("omega_best:", best_omega)
        print(f"pest: {best_pest:0.4f} +- {sigma_P:0.6f}, global uncertainty: {like_P:0.4f}")
        # print("coeffs (mu, d, a, phi):", coeffs)  # [mu, d, a, phi]
        # print("cov (mu, d, a):\n", cov)
        # print(f"execution took {time.time() - start:0.2f} seconds")

    return dict({'coeffs':coeffs, 'period':best_pest, 'cov':cov, 'variance':sigma_P, 'posterior':like_P, 'next_best':next_best, 'rss':rss})

def plot_lc(lc_in, ax=None, mag=True, title=None, err_cutoff = 0.5, bare=True, filters = ['u', 'g', 'r', 'i', 'z', 'y']): #bare tells if you've put in just the light curve, or the whole gen_lc
    if ax is None:
        fig, ax = plt.subplots()
    if bare:
        lc = lc_in
    else:
        lc = lc_in.lightcurve
    for band in filters:
        this = lc[lc['band'] == band] #band selection
        ylabel = 'mag' if mag else 'flux (nJy)'
        if title is None:
            title = f"Lightcurve {lc.id} at ({lc.ra:.2f}, {lc.dec:.2f})"
        ax.errorbar(this.midpointMjdTai, this.psfMag, yerr=this.psfMagErr, color=filter_colors[band], label=band, ls=' ', marker='.') #err_mask was on these
        ax.set(xlabel = 'MJD', ylabel = ylabel, title=title)
        ax.legend()
    if mag:
        ax.invert_yaxis()

#period in days
def phase_fold_lc(lc_in, period, ax=None, mag=True, title=None, coeff=0, err_cutoff = 0.5, bare=True, filters = ['u', 'g', 'r', 'i', 'z', 'y']):
    if ax is None:
        fig, ax = plt.subplots()
    if bare:
        lc = lc_in
    else:
        lc = lc_in.lightcurve
        
    for band in filters:
        this = lc[lc['band'] == band] #band selection
        ylabel = 'mag' if mag else 'flux (nJy)'
        if title is None:
            title = f"Phase-Folded Lightcurve {lc.id} with period {period:0.3f} days at ({lc.ra:.2f}, {lc.dec:.2f})"
        phase = np.mod(this.midpointMjdTai + coeff*period, period)/period
        ax.errorbar(phase, this.psfMag, yerr=this.psfMagErr, color=filter_colors[band], label=band, ls=' ', marker='.') #err_mask was on these
        ax.legend()
        ax.set(xlabel = 'phase', ylabel = ylabel, title=title)
    if mag:
        ax.invert_yaxis()

        
def extract_multiband_results(coeffs: np.ndarray, cov: np.ndarray, band_names: list = None) -> Dict[str, Any]:
    """
    Extract and organize multiband fitting results in a user-friendly format.
    
    Parameters
    ----------
    coeffs : np.ndarray
        Output from ComputeCoeffsAndCov_with_offset_numba
    cov : np.ndarray
        Covariance matrix from ComputeCoeffsAndCov_with_offset_numba
    band_names : list, optional
        Names of bands (e.g., ['u', 'g', 'r', 'i', 'z']).
        If None, bands are labeled as 'Band_0', 'Band_1', etc.
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'mu': baseline magnitude with error
        - 'amplitude': amplitude with error
        - 'phase': phase (no error, nonlinear parameter)
        - 'dust': dust parameter with error (if fitted)
        - 'band_offsets': dict of band names -> offset values with errors
        - 'chi2_red': reduced chi-squared
        - 'correlation_matrix': Pearson correlation matrix
    """
    n_bands = len(coeffs) - 3
    
    if band_names is None:
        band_names = [f'Band_{i}' for i in range(n_bands + 1)]
    
    mu = coeffs[0]
    d = coeffs[1]
    a = coeffs[2]
    phi = coeffs[3]
    band_offsets = coeffs[4:]
    
    # Extract errors (diagonal of covariance)
    mu_err = np.sqrt(cov[0, 0])
    d_err = np.sqrt(cov[1, 1])
    a_err = np.sqrt(cov[2, 2])
    
    band_offset_errs = np.array([np.sqrt(cov[3 + i, 3 + i]) for i in range(n_bands-1)])
    
    # Build results dict
    results = {
        'mu': {'value': mu, 'error': mu_err},
        'amplitude': {'value': a, 'error': a_err},
        'phase': phi,  # nonlinear, no error from linear covariance
        'dust': {'value': d, 'error': d_err},
        'band_offsets': {},
        'covariance_matrix': cov,
        'correlation_matrix': cov2corr(cov),
    }
    
    # Add band offsets (first n_bands-1 are free, last is reference = 0)
    for i in range(n_bands - 1):
        results['band_offsets'][band_names[i]] = {
            'value': band_offsets[i],
            'error': band_offset_errs[i]
        }
    results['band_offsets'][band_names[n_bands - 1]] = {
        'value': 0.0,
        'error': 0.0,
        'note': 'reference band (fixed)'
    }
    
    return results

def cov2corr(cov: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr

def print_multiband_results(results: Dict[str, Any], decimals: int = 4) -> None:
    """
    Pretty-print multiband fitting results.
    
    Parameters
    ----------
    results : dict
        Output from extract_multiband_results
    decimals : int
        Number of decimal places to display
    """
    fmt = f"{{:.{decimals}f}}"
    
    print("\n" + "="*60)
    print("MULTIBAND TEMPLATE FITTING RESULTS")
    print("="*60)
    
    # Basic parameters
    mu = results['mu']
    print(f"\nBaseline magnitude (μ):")
    print(f"  {fmt.format(mu['value'])} ± {fmt.format(mu['error'])}")
    
    amp = results['amplitude']
    print(f"\nAmplitude (a):")
    print(f"  {fmt.format(amp['value'])} ± {fmt.format(amp['error'])}")
    
    print(f"\nPhase (φ):")
    print(f"  {fmt.format(results['phase'])}")
    
    dust = results['dust']
    print(f"\nDust parameter (d):")
    print(f"  {fmt.format(dust['value'])} ± {fmt.format(dust['error'])}")
    
    # Band offsets
    print(f"\nBand offsets (relative to reference band):")
    for band, offset_dict in results['band_offsets'].items():
        if 'note' in offset_dict:
            print(f"  {band:12s}: {fmt.format(offset_dict['value']):8s} (reference)")
        else:
            print(f"  {band:12s}: {fmt.format(offset_dict['value'])} ± {fmt.format(offset_dict['error'])}")
    
    print("\n" + "="*60)

def reorder_template_for_lc(tem_py, lc_bands):
    """
    Return a copy of tem_py reordered (and subset) to match lc_bands order.

    Parameters
    ----------
    tem_py : dict
        Template dictionary with keys:
         - 'bands' : list of band names (original template order)
         - 'templates' : ndarray (n_bands, n_phase)
         - 'templatesd' : ndarray (n_bands, n_phase)
         - 'betas' : ndarray shape (3, n_bands) or (n_bands, 3)
         - 'template_funcs' : list of callables in original band order
         - 'templated_funcs' : list of callables in original band order
         - 'dust', 'model_error' : dict band->value
         - other keys preserved
    lc_bands : sequence
        Unique bands from your light curve in the desired order (e.g. list(pd.unique(lc['band'])))

    Returns
    -------
    tem_new : dict
        New template dict with bands ordered exactly as lc_bands (and template entries subsetted).
    """
    tem = dict(tem_py)  # shallow copy to start
    tem_bands = list(tem["bands"])
    # ensure every lc band exists in template
    missing = [b for b in lc_bands if b not in tem_bands]
    if missing:
        raise ValueError(f"These bands are in the light curve but not in the template: {missing}")

    # Build new band list in lc order; also drop template bands not in lc
    keep = [b for b in lc_bands if b in tem_bands]

    # Map old indices -> new order
    old_idx = [tem_bands.index(b) for b in keep]

    # Reorder templates (n_bands, n_phase) by rows
    templates = np.asarray(tem["templates"])
    templatesd = np.asarray(tem["templatesd"])
    if templates.shape[0] != len(tem_bands):
        # try transpose if orientation was (n_phase, n_bands)
        if templates.shape[1] == len(tem_bands):
            templates = templates.T
            templatesd = templatesd.T
        else:
            raise ValueError("Unexpected templates shape vs tem['bands'] length: "
                             f"{templates.shape} vs {len(tem_bands)}")

    templates_new = templates[old_idx, :]
    templatesd_new = templatesd[old_idx, :]

    # Reorder betas: expected shape (3, n_bands). If shape matches (n_bands,3) transpose
    betas = np.asarray(tem["betas"])
    if betas.ndim == 2 and betas.shape[0] == len(tem_bands) and betas.shape[1] == 3:
        betas = betas.T  # to shape (3, n_bands)
    if betas.shape[1] != len(tem_bands):
        raise ValueError("Unexpected betas shape; expected (3, n_bands) or (n_bands,3)")

    betas_new = betas[:, old_idx]  # columns reordered

    # Reorder template_funcs and templated_funcs (lists)
    tf = list(tem["template_funcs"])
    tfd = list(tem["templated_funcs"])
    tf_new = [tf[i] for i in old_idx]
    tfd_new = [tfd[i] for i in old_idx]

    # Reorder dust and model_error dicts
    dust_new = {b: float(tem["dust"][b]) for b in keep}
    me_new = {b: float(tem["model_error"][b]) for b in keep}

    # Create new template dict (copy other keys that remain valid)
    tem_new = dict(tem_py)  # start with original copy
    tem_new["bands"] = keep
    tem_new["templates"] = templates_new
    tem_new["templatesd"] = templatesd_new
    tem_new["betas"] = betas_new
    tem_new["template_funcs"] = tf_new
    tem_new["templated_funcs"] = tfd_new
    tem_new["dust"] = dust_new
    tem_new["model_error"] = me_new

    # If abs_mag uses tem['betas'] it will now use the reordered betas; no further change needed.
    return tem_new

# ---------- Template I/O functions ----------
"""
Utilities to save/load the template dict produced by convert_tem_r_to_py.

Two approaches:
- save_template/save_template_dir: robust portable save of numeric components (npz + json).
- quick_save_template / quick_load_template: single-file save using joblib (less portable).
"""
import json
import numpy as np
from pathlib import Path

try:
    from scipy.interpolate import interp1d
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Small helper to build interpolation callables (same as conversion helper)
def _make_interp_func(temp_time, y):
    x = np.asarray(temp_time, dtype=float)
    y = np.asarray(y, dtype=float)
    if _HAS_SCIPY:
        interp = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)
        def f(v):
            v = np.asarray(v, dtype=float) % 1.0
            return interp(v)
    else:
        # numpy.interp fallback; assumes x sorted and v in [0,1)
        def f(v):
            v = np.asarray(v, dtype=float) % 1.0
            return np.interp(v, x, y)
    return f

def _build_template_funcs(temp_time, templates, templatesd, bands):
    # templates: (n_bands, M), templatesd: (n_bands, M)
    n_bands = len(bands)
    template_funcs = []
    templated_funcs = []
    for bi in range(n_bands):
        template_funcs.append(_make_interp_func(temp_time, templates[bi, :]))
        templated_funcs.append(_make_interp_func(temp_time, templatesd[bi, :]))
    return template_funcs, templated_funcs

def _make_abs_mag_func(betas):
    # betas shape expected (3, n_bands)
    bet = np.asarray(betas)
    if bet.shape[0] != 3 and bet.shape[1] == 3:
        bet = bet.T
    def abs_mag(periods, tem=None):
        p = np.asarray(periods, dtype=float)
        if p.ndim == 0:
            p = p[None]
        logp = np.log10(p) + 0.2
        X = np.column_stack([np.ones_like(p), logp, logp**2])  # (n_periods, 3)
        return X.dot(bet)  # (n_periods, n_bands)
    return abs_mag

def save_template_dir(tem_py, path):
    """
    Save the numeric parts of tem_py into `path` (a directory).
    Creates:
      path/components.npz  -- numpy arrays: betas, templates, templatesd, temp_time
      path/meta.json       -- bands, dust, model_error
    Use load_template_dir(path) to re-create tem_py with callables.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    # arrays
    arrs = {
        "betas": np.asarray(tem_py["betas"]),
        "templates": np.asarray(tem_py["templates"]),
        "templatesd": np.asarray(tem_py["templatesd"]),
        "temp_time": np.asarray(tem_py["temp_time"]),
    }
    np.savez_compressed(p / "components.npz", **arrs)
    # metadata (convert numeric dicts to plain lists/values)
    meta = {
        "bands": list(tem_py["bands"]),
        "dust": {b: float(tem_py["dust"][b]) for b in tem_py["bands"]},
        "model_error": {b: float(tem_py["model_error"][b]) for b in tem_py["bands"]}
    }
    with open(p / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return p.resolve()

def load_template_dir(path):
    """
    Load from directory produced by save_template_dir and reconstruct tem_py dict
    including template_funcs and templated_funcs and abs_mag callable.
    """
    p = Path(path)
    npz = np.load(p / "components.npz", allow_pickle=False)
    betas = npz["betas"]
    templates = npz["templates"]
    templatesd = npz["templatesd"]
    temp_time = npz["temp_time"]
    with open(p / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    bands = meta["bands"]
    dust = meta["dust"]
    model_error = meta["model_error"]
    # ensure shapes orientation: templates (n_bands, M)
    # reconstruct functions
    template_funcs, templated_funcs = _build_template_funcs(temp_time, templates, templatesd, bands)
    abs_mag = _make_abs_mag_func(betas)
    tem_py = {
        "bands": bands,
        "dust": dust,
        "model_error": model_error,
        "betas": betas,
        "templates": templates,
        "templatesd": templatesd,
        "temp_time": temp_time,
        "template_funcs": template_funcs,
        "templated_funcs": templated_funcs,
        "abs_mag": abs_mag,
    }
    return tem_py

# Quick one-file save (less portable): joblib / cloudpickle
def quick_save_template(tem_py, filename):
    """
    Save entire tem_py dict (including callables) using joblib (cloudpickle).
    This is convenient but less portable than the directory approach.
    """
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("Install joblib (pip install joblib) to use quick_save_template") from e
    joblib.dump(tem_py, filename, compress=3)
    return Path(filename).resolve()

def quick_load_template(filename):
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("Install joblib (pip install joblib) to use quick_load_template") from e
    return joblib.load(filename)

# Example usage:
# save_template_dir(tem_py, "my_template")
# tem_py2 = load_template_dir("my_template")
# OR quick_save_template(tem_py, "tem.pkl"); tem_py2 = quick_load_template("tem.pkl"