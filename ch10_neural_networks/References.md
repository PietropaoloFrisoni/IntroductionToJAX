# Chapter 10: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [Flax NNX basics](https://flax.readthedocs.io/en/latest/nnx_basics.html)
**The introduction to the Flax NNX API used in this chapter: modules, parameters as attributes, and how state is handled.**

NNX is the current Flax API and the one this course teaches. A lot of material online still uses the older Linen API, which looks quite different, so check which one you are reading.

---

## Reference 2: [Flax documentation](https://flax.readthedocs.io/en/latest/index.html)
**The full documentation: guides, examples and the API reference.**

Useful once you go past what the chapter covers, particularly the guides on saving and restoring model state.

---

## Reference 3: [Optax](https://optax.readthedocs.io/en/latest/index.html)
**The gradient-processing and optimization library used for the training loop.**

Optax composes transformations rather than hiding an optimizer behind one object. Once that clicks, gradient clipping, scheduling and weight decay all become the same kind of thing.

---

## Reference 4: [Optax optimizers](https://optax.readthedocs.io/en/latest/api/optimizers.html)
**The catalogue of ready-made optimizers, with the formula each one implements.**

Worth a look even if you only ever use `adam`, because the entries show how each is assembled from the smaller pieces.

---

## Reference 5: [Training a simple neural network](https://docs.jax.dev/en/latest/notebooks/Neural_Network_and_Data_Loading.html)
**The official end-to-end example: a network written in pure JAX, trained on real data, with no framework involved.**

Very close to the first half of this chapter. Good as a second pass over the same ideas, with a complete data pipeline attached.

---

## Reference 6: [jax.tree_util.tree_map](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.tree_map.html)
**Reference for applying a function across every leaf of a nested structure.**

This one line is what makes a gradient step work on an arbitrarily nested parameter tree. It is the quiet centre of the pure-JAX training loop.

---

## Reference 7: [Stateful computations](https://docs.jax.dev/en/latest/stateful-computations.html)
**How to handle state (parameters, optimizer state, counters) inside a functional model.**

The question this chapter runs into as soon as training starts, explained in general terms rather than through one framework's answer to it.

---
