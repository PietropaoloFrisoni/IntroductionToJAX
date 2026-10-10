# Chapter 2: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [Just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
**The official walkthrough of `jax.jit`: what gets traced, what gets cached, and the conditions under which a function is recompiled.**

Covers the same ground as this chapter from a different angle. The section on caching is the one worth re-reading.

---

## Reference 2: [jax.jit](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)
**Full API reference, including `static_argnums`, `static_argnames`, `donate_argnums` and the compiler options.**

`static_argnums` is the argument that solves most real recompilation problems, and the documentation here is more precise than any tutorial on it.

---

## Reference 3: [Ahead-of-time compilation](https://docs.jax.dev/en/latest/aot.html)
**How to split the stages that `jit` normally performs together: trace, lower, compile, and only then execute.**

This is the page behind `.lower()` and `.compile()`, which we use in this chapter to look at what the compiler received. It becomes the backbone of Chapters 3 and 4.

---

## Reference 4: [Persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
**How to make compiled executables survive across processes, so the first call is not slow every time you restart.**

Not needed for the course, but genuinely useful once you work on something real: it turns a recurring wait into a one-off.

---

## Reference 5: [jax.block_until_ready](https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html)
**API reference for the function that forces asynchronous dispatch to complete.**

Short page, but the one to point at whenever someone benchmarks JAX and reports an impossibly fast result.

---

## Reference 6: [JAX Errors](https://docs.jax.dev/en/latest/errors.html)
**The catalogue of JAX's own exception types, each with the cause and the usual fix.**

`TracerBoolConversionError` and `ConcretizationTypeError` are the two you will meet first, and they are both explained here better than the traceback explains them.

---

## Reference 7: [Debugging runtime values](https://docs.jax.dev/en/latest/debugging.html)
**The official tools for inspecting values inside compiled code: `jax.debug.print`, `jax.debug.breakpoint` and `jax.disable_jit`.**

Relevant to the section on side effects: once you know why a plain `print` fires only at trace time, these are what you use instead.

---
