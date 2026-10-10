# Chapter 9: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [jax.numpy module](https://docs.jax.dev/en/latest/jax.numpy.html)
**The complete API reference for the NumPy-compatible interface, with notes marking where behaviour differs from NumPy.**

The deviation notes are the valuable part. When something behaves unexpectedly, the explanation is usually right there in the entry for the function.

---

## Reference 2: [Pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html)
**Why JAX uses explicit keys instead of global state, how splitting works, and what it guarantees about reproducibility.**

This chapter's most important reference. The design looks like an inconvenience until you see what it buys you under `jit` and `vmap`, and this page makes that argument properly.

---

## Reference 3: [jax.random.key](https://docs.jax.dev/en/latest/_autosummary/jax.random.key.html)
**Reference for creating a key, and the difference from the older `PRNGKey`.**

Worth reading because the older form appears in a great deal of existing code, including code you will find in tutorials.

---

## Reference 4: [jax.random.split](https://docs.jax.dev/en/latest/_autosummary/jax.random.split.html)
**Reference for deriving independent keys from an existing one.**

The operation the whole design rests on. Reusing a key instead of splitting it is the single most common randomness bug in JAX.

---

## Reference 5: [The .at[] indexed update syntax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html)
**The functional replacement for in-place assignment, with the full set of operations: `set`, `add`, `multiply`, `min`, `max` and the rest.**

The `mode` argument is the part to notice: it controls what happens with out-of-bounds indices, which is exactly the silent difference this chapter demonstrates.

---

## Reference 6: [Type promotion semantics](https://docs.jax.dev/en/latest/type_promotion.html)
**The complete table of what dtype results from combining any two, and how JAX's rules differ from NumPy's.**

The promotion table explains results that otherwise look arbitrary, especially once Python scalars are mixed with arrays.

---

## Reference 7: [jax.numpy.float64 and 64-bit precision](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float64.html)
**Reference entry noting that double precision requires enabling x64 explicitly, and what happens if you do not.**

Directly relevant to the 32-versus-64-bit section. The default catches out anyone arriving from scientific computing, where double precision is simply assumed.

---

## Reference 8: [Common Gotchas in JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
**The collection of surprising behaviours: pure functions, immutability, out-of-bounds indexing, randomness and control flow.**

Read it again here. After this chapter every item on the list reads as an obvious consequence rather than a quirk, which is a good way to check your own understanding.

---
