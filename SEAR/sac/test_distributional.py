from sac.distributional import project
from sac.config import DistConfig
import jax.numpy as jnp

dint = DistConfig(11, 0, 10)
dpm = DistConfig(101, -100, 100)


def test_nop():
    d = dint
    start = jnp.arange(d.n_atoms)
    start /= jnp.sum(start)

    res = project(start, add=0, scale=1, dist=d)
    assert jnp.allclose(start, res)


def test_add():
    d = dint
    start = jnp.array([0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0])
    start /= jnp.sum(start)

    offset = 1.5

    res = project(start, add=offset, scale=1, dist=d)
    assert jnp.allclose(d.expected_value(res), d.expected_value(start) + offset)


def test_mult():
    d = dint
    start = jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    start /= jnp.sum(start)

    factor = 2

    res = project(start, add=0, scale=factor, dist=d)
    e_res = d.expected_value(res)
    e_expected = d.expected_value(start) * factor
    assert jnp.allclose(e_expected, e_res)


def test_mult_0():
    d = dpm
    start = jnp.zeros_like(d.atoms).at[20].set(1)
    start /= jnp.sum(start)

    res = project(start, add=0, scale=0, dist=d)
    e_res = d.expected_value(res)
    assert jnp.allclose(0, e_res)


def test_mult_factor():
    d = dint
    start = jnp.array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    start /= jnp.sum(start)

    factor = 1.5
    offset = 1.5

    res = project(start, add=offset, scale=factor, dist=d)
    assert jnp.allclose(
        d.expected_value(res), offset + d.expected_value(start) * factor
    )
