# Chapter 5: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [JAX Primitives](https://docs.jax.dev/en/latest/jax-primitives.html)
**What a primitive is in JAX's own terms, and the set of rules a primitive must provide for each transformation that may be applied to it.**

The official statement of what we build by hand in this chapter. The table of rules is a good checklist when your own primitive works under `jit` but not under `vmap`.

---

## Reference 2: [How JAX primitives work](https://docs.jax.dev/en/latest/notebooks/How_JAX_primitives_work.html)
**A notebook that defines a new primitive step by step, registering the implementation, abstract evaluation, lowering, differentiation and batching rules.**

The closest thing in the documentation to this chapter, and useful as a second worked example once you have followed ours.

---

## Reference 3: [Autodidax: JAX core from scratch](https://docs.jax.dev/en/latest/autodidax.html)
**A rebuild of JAX's core from first principles (tracing, jaxprs, transformations and primitives) in plain Python, in a single readable document.**

This is the deepest page in the entire JAX documentation, and my favourite. It is long, and it is the one thing I would point to if you want to go further than this course does.

---

## Reference 4: [Custom derivative rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
**The official guide to `custom_jvp` and `custom_vjp`: when the automatic derivative is wrong, numerically unstable, or simply slower than one you can write yourself.**

The practical counterpart to defining a primitive. In real code you reach for these far more often than for a new primitive.

---

## Reference 5: [jax.custom_vjp](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html)
**API reference for reverse-mode custom derivatives, with the `fwd`/`bwd` function signatures.**

Keep it open the first time you write one: the split between the forward pass and its residuals is easy to get subtly wrong.

---

## Reference 6: [jax.custom_jvp](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html)
**API reference for forward-mode custom derivatives.**

Simpler than `custom_vjp`, and the right choice when the function is cheap and you only need the tangent rule.

---

## Reference 7: [jax.lax module](https://docs.jax.dev/en/latest/jax.lax.html)
**The lower-level operation library that `jax.numpy` is built on: the primitives themselves, with their exact semantics.**

When you want to know what a `jax.numpy` call really does, the answer is usually one `jax.lax` call away.

---
