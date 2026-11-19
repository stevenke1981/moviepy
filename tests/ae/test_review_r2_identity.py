"""Stable layer identities keep random streams independent of names and order."""

import json

import numpy as np

from moviepy.ae import Composition, RenderContext, Transform
from moviepy.ae.layers import SolidLayer
from moviepy.ae.properties import Expression, Property


def test_same_name_wiggle_layers_keep_independent_stable_displacements():
    """Renaming and moving layers must not change their random displacement."""
    comp = Composition(size=(64, 64), fps=30, duration=2)
    layers = [comp.add_solid("Solid") for _ in range(2)]
    for layer in layers:
        layer.transform = Transform(
            position=Property(
                (0, 0), value_type="vec2", expression=Expression("wiggle(2, 20)")
            )
        )
    context = RenderContext(rng_seed=19)
    identities = [layer.id for layer in layers]
    matrices = [layer.local_matrix(0.37, context) for layer in layers]
    assert identities[0] != identities[1]
    assert not np.array_equal(matrices[0][:2, 2], matrices[1][:2, 2])
    comp.move_layer(layers[0], 1)
    layers[0].name = "renamed"
    for layer, identity, matrix in zip(layers, identities, matrices):
        assert layer.id == identity
        np.testing.assert_array_equal(layer.local_matrix(0.37, context), matrix)


def test_layer_identity_roundtrips_through_json():
    """A restored identity reproduces the same random stream."""
    layer = SolidLayer("Solid", size=(2, 2))
    restored = SolidLayer("renamed", id=json.loads(json.dumps(layer.id)), size=(2, 2))
    assert restored.id == layer.id
    context = RenderContext(t=0.5, rng_seed=3)
    np.testing.assert_array_equal(
        context.rng_for(layer.id).random(10),
        context.rng_for(restored.id).random(10),
    )
