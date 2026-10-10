# Chapter 1: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [Understanding Jaxprs](https://docs.jax.dev/en/latest/jaxpr.html)
**The official description of the jaxpr language: its grammar, how to read the equations, and how constants and literals appear in it.**

This is the page that matches this chapter most closely. Read it after the lecture rather than before: the notation makes far more sense once you have watched one being produced.

---

## Reference 2: [jax.make_jaxpr](https://docs.jax.dev/en/latest/_autosummary/jax.make_jaxpr.html)
**API reference for the function we use throughout the chapter to inspect what JAX built from our Python.**

Note the `static_argnums` argument. It becomes relevant in Chapter 2, and seeing it here first makes that discussion easier.

---

## Reference 3: [Key Concepts: tracing and transformations](https://docs.jax.dev/en/latest/key-concepts.html)
**Explains what a tracer is, why your Python function runs on abstract values rather than numbers, and what that implies for anything that depends on a concrete value.**

The short section on tracers is the one to read. Almost every confusing JAX error traces back to that one idea.

---

## Reference 4: [Control Flow and Logical Operators](https://docs.jax.dev/en/latest/control-flow.html)
**Why Python `if` and `while` on traced values do not do what you expect, and the structured alternatives JAX provides instead.**

This answers the question the chapter raises when Python control flow meets a dynamic value. The structured primitives are covered properly in Chapter 6, but the explanation of *why* is here.

---

## Reference 5: [Pytrees](https://docs.jax.dev/en/latest/pytrees.html)
**The container abstraction JAX uses to let transformations work on nested structures of arrays: lists, dicts, tuples and custom classes.**

Pytrees look like a minor convenience at first and turn out to be central by Chapter 10, where an entire neural network is one.

---

## Reference 6: [Common Gotchas in JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
**A notebook collecting the behaviours that most often surprise people coming from NumPy: pure functions, in-place updates, out-of-bounds indexing, randomness and control flow.**

I would read this once early and again after Chapter 9. The first time it reads as a list of warnings. The second time, every item is an obvious consequence of something you now understand.

---
