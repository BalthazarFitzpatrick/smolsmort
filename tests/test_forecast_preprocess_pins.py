"""pins ridge and mlp outputs to the last digit across the shared preprocessing refactor"""

import numpy as np
import pytest

from smolsmort.forecast.families import get_family

RIDGE_PREDICTIONS = (
    "0x1.d57b304cfe454p+0",
    "0x1.37236c2b3fb0fp+3",
    "-0x1.7a4332cc681c1p+0",
    "-0x1.480dc9a80752ap+3",
    "0x1.2694d90244fddp+1",
    "-0x1.43bdcc9045751p+1",
    "0x1.168aa3152fc73p+2",
    "-0x1.02e598468b2fdp+1",
    "0x1.7362b3984c99dp+3",
    "0x1.851d30cfb5233p+1",
    "-0x1.e22e06698f6abp+0",
    "0x1.e325c20273b14p+2",
    "0x1.88b25bc95c6b9p+0",
    "0x1.40557fe54cac7p+3",
    "-0x1.7b72f994be694p+2",
    "0x1.a45f4c3239caap+1",
    "-0x1.be0d3f907725ep-1",
    "0x1.7006d19e1c16ep+3",
    "-0x1.51ca189b6ba15p+0",
    "-0x1.4985bb619d1e0p+2",
    "0x1.6c128b910b68ap-3",
    "-0x1.27d4cab2c0c6cp-4",
    "0x1.0aab323beabdap-2",
    "-0x1.1ca284b0e3a55p+1",
    "-0x1.e575d31da1fc6p+2",
)

MLP_PREDICTIONS = (
    "0x1.04263abae5bcap+1",
    "0x1.0062d81149cd8p+2",
    "-0x1.3e41d8adcfb70p-1",
    "-0x1.a6907e52cccf8p+2",
    "0x1.1715ec38b06d4p+0",
    "-0x1.1c7aaae8d1c99p+1",
    "0x1.84dbe4e94c7fap+1",
    "-0x1.6be03d38eb43ap+0",
    "0x1.27af0e8da156ep+2",
    "0x1.314f40791528ap+1",
    "-0x1.754790ed1370ap+0",
    "0x1.c169f3efb5371p+1",
    "0x1.9f0aa393e9366p-1",
    "0x1.07dd7da6887c7p+2",
    "-0x1.ec59612eade12p+1",
    "0x1.6f922bb9c329bp+0",
    "-0x1.10ab7d9b25530p-1",
    "0x1.47481ce579e7dp+2",
    "-0x1.1716dd969fe3ap-1",
    "-0x1.00bb93011c721p+1",
    "0x1.3156520ed7ba3p-2",
    "-0x1.6f46617f36c87p-2",
    "0x1.27b59813dd758p-4",
    "-0x1.d22304761ed2ap-1",
    "-0x1.06d9a740dcf95p+2",
)

MLP_VALIDATION = (
    "0x1.f497d80000000p-1",
    "0x1.e5b23e0000000p-1",
    "0x1.d733aa0000000p-1",
    "0x1.c8beb20000000p-1",
    "0x1.bb9ec00000000p-1",
    "0x1.af88160000000p-1",
    "0x1.a45bca0000000p-1",
    "0x1.9a41b80000000p-1",
    "0x1.90757e0000000p-1",
    "0x1.862bd60000000p-1",
    "0x1.7badda0000000p-1",
    "0x1.719b9e0000000p-1",
    "0x1.6811b60000000p-1",
    "0x1.5ecde60000000p-1",
    "0x1.553ca60000000p-1",
    "0x1.4bfd0e0000000p-1",
    "0x1.42bba60000000p-1",
    "0x1.39de4e0000000p-1",
    "0x1.3030080000000p-1",
    "0x1.270eb00000000p-1",
    "0x1.1e0f860000000p-1",
    "0x1.15311e0000000p-1",
    "0x1.0c57240000000p-1",
    "0x1.03b32a0000000p-1",
    "0x1.f7317e0000000p-2",
    "0x1.e696720000000p-2",
    "0x1.d7418a0000000p-2",
    "0x1.c8d1000000000p-2",
    "0x1.ba3cd00000000p-2",
    "0x1.ac7a560000000p-2",
)

RIDGE_DESIGN = ("float64", (25, 13), "0x1.809532751c21cp+5")
MLP_DESIGN = ("float32", (25, 13), "0x1.8095326f50000p+5")
MLP_EPOCHS = 30
TYPES = ["q", "q", "q", "q", "q", "c"]


def make_data():
    rng = np.random.default_rng(7)
    n = 160
    x = rng.normal(size=(n, 6)) * [1, 5, 0.1, 1, 1, 1]
    x[:, 3] = 2.5
    x[:, 4] = np.nan
    x[:, 5] = rng.integers(0, 4, n).astype(float)
    x[rng.random(n) < 0.1, 1] = np.nan
    x[rng.random(n) < 0.1, 5] = np.nan
    y = x[:, 0] * 2 + np.nan_to_num(x[:, 1]) + (x[:, 5] == 2) * 3 + rng.normal(size=n) * 0.1
    probe = x[:25].copy()
    # one unseen category level
    probe[0, 5] = 9.0
    return x, y, probe


def hexes(values) -> tuple[str, ...]:
    return tuple(float(v).hex() for v in values)


def design_pin(design) -> tuple[str, tuple, str]:
    return (
        str(design.dtype),
        design.shape,
        float(np.asarray(design, dtype=np.float64).sum()).hex(),
    )


def test_ridge_outputs_are_unchanged():
    x, y, probe = make_data()
    fitted = get_family("ridge").fit(x, y=y, feature_types=TYPES, seed=3, params={"alpha": 1.0})
    assert hexes(fitted.predict(probe)) == RIDGE_PREDICTIONS
    assert design_pin(fitted.transform(probe)) == RIDGE_DESIGN


def test_mlp_outputs_are_unchanged():
    pytest.importorskip("torch")
    x, y, probe = make_data()
    fitted = get_family("mlp").fit(
        x,
        y=y,
        feature_types=TYPES,
        seed=3,
        max_rounds=30,
        patience=4,
        params={"n_layers": 1, "width": 8, "batch_size": 32},
    )
    assert hexes(fitted.predict(probe)) == MLP_PREDICTIONS
    assert design_pin(fitted.prep.transform(probe)) == MLP_DESIGN
    assert fitted.epochs == MLP_EPOCHS
    assert hexes(fitted.eval_history["validation"]) == MLP_VALIDATION
