from dataclasses import dataclass
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped, Scalar
from typeguard import typechecked as typechecker
from typing import Union

Floaty = Union[float, Scalar]


@dataclass(frozen=True)
class DistConfig:
    n_atoms: int
    v_min: float
    v_max: float

    @property
    def delta_z(self) -> float:
        return (self.v_max - self.v_min) / (self.n_atoms - 1)

    @property
    def atoms(self) -> Float[Array, " n_atoms"]:
        return jnp.linspace(self.v_min, self.v_max, self.n_atoms)

    def expected_value(self, weights: Float[Array, " n_atoms"]) -> Floaty:
        return jnp.sum(weights * self.atoms)

    def quartile(self, weights: Float[Array, " n_atoms"], q: float) -> Floaty:
        # computes the given quartile, interpolating linearly between buckets
        assert 0 <= q <= 1
        cum = jnp.cumsum(weights)
        ri = jnp.searchsorted(cum, q)
        li = jnp.maximum(ri - 1, 0)
        cl = cum[li]
        cr = cum[ri]
        # cl < q <= cr
        d = cr - cl
        alpha = (q - cl) / d
        alpha = jnp.nan_to_num(alpha, nan=0)  # if li == ri
        # 0 < alpha <= 1
        return (1 - alpha) * self.atoms[li] + alpha * self.atoms[ri]

    @jaxtyped(typechecker=typechecker)
    def dirac(self, x: Floaty) -> Float[Array, " n_atoms"]:
        """maps a dirac dist with expected value x into this categorical dist"""
        if type(x) is float:
            assert self.v_min <= x <= self.v_max
        # create a dummy weights vector with norm 1 and all weight in the first atom
        init = jnp.zeros(self.n_atoms).at[0].set(1)
        init_exp = self.expected_value(init)

        # rescale to target x
        shift = x - init_exp
        result = project(init, add=shift, scale=1, dist=self)

        return result


@jaxtyped(typechecker=typechecker)
def project(
    weights: Float[Array, " n_atoms"], add: Floaty, scale: Floaty, dist: DistConfig
) -> Float[Array, " n_atoms"]:
    m = jnp.zeros_like(weights)
    tz = add + scale * dist.atoms
    tz = jnp.clip(tz, min=dist.v_min, max=dist.v_max)
    b = (tz - dist.v_min) / dist.delta_z

    # ensure that b is within bound, even under floating point errors
    b = jnp.clip(b, 0, dist.n_atoms - 1)

    l = jnp.floor(b).astype(jnp.int32)  # noqa: E741
    u = jnp.ceil(b).astype(jnp.int32)

    # make sure l != u, even if b is an integer

    l = jnp.where((u > 0) & (l == u), l - 1, l)  # noqa: E741
    u = jnp.where((l < (dist.n_atoms - 1)) & (l == u), u + 1, u)

    m = m.at[l].add(weights * (u - b))
    m = m.at[u].add(weights * (b - l))
    m = m + 1e-10
    return m / jnp.sum(m)
