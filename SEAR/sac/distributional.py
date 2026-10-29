from sac.config import DistConfig
from jaxtyping import jaxtyped, Float, Array
from typeguard import typechecked as typechecker
import jax.numpy as jnp

from sac.utils import Floaty


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
