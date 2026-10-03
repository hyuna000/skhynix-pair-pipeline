import warnings
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from src import unitroot as ur, hedge

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)
Y = np.cumsum(rng.standard_normal(1440))

def test_adf_matches_statsmodels():
    for k in (0, 3, 8):
        assert np.isclose(ur.adf_select(Y, kmax=k, fixed_k=k)["stat"], adfuller(Y, maxlag=k, autolag=None)[0])

def test_kpss_matches_statsmodels():
    x = rng.standard_normal(1440)
    assert np.isclose(ur.kpss_stat(x)["stat"], kpss(x, regression="c", nlags="auto")[0])
    assert np.isclose(ur.kpss_stat(Y)["stat"], kpss(Y, regression="c", nlags="auto")[0])

def test_hedge_recovers_b():
    x = np.cumsum(rng.standard_normal(1440)) * 0.001
    y = 0.9 * x + np.log(0.1) + 0.0005 * rng.standard_normal(1440)
    for m in ("ols", "tls", "dols"):
        assert abs(hedge.estimate(y, x, m)[1] - 0.9) < 0.05
