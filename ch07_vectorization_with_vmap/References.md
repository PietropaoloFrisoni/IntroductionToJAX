# Chapter 7: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [Automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html)
**The official introduction to `vmap`: writing a function for one example and applying it to a batch, with `in_axes` and `out_axes` explained through examples.**

Covers the same material as this chapter and is short enough to read in one sitting. The manual-vectorization comparison at the start makes the point better than any description.

---

## Reference 2: [jax.vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
**Full API reference, including `in_axes`, `out_axes`, `axis_name` and `axis_size`.**

The precise rules for how `in_axes` maps onto a nested structure are here, and they are easier to look up than to deduce.

---

## Reference 3: [The Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
**Among much else, the derivation of per-example gradients by composing `vmap` with `grad`.**

Directly relevant to the per-sample gradients section. Seeing that per-example gradients are one line rather than a special algorithm is the moment `vmap` clicks for most people.

---

## Reference 4: [Pytrees](https://docs.jax.dev/en/latest/pytrees.html)
**How transformations map over nested containers, which is what `in_axes` is really specifying when the argument is not a bare array.**

Once a function takes a dictionary of parameters, `in_axes=None` on that dictionary is the thing you need, and this page explains why it works.

---
