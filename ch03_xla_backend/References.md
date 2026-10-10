# Chapter 3: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [XLA](https://openxla.org/xla)
**The home page of the compiler JAX targets, maintained as part of the OpenXLA project.**

Useful mostly to see that XLA is a project in its own right, with users beyond JAX. The structure of the documentation also tells you what the compiler considers its own responsibilities.

---

## Reference 2: [XLA Architecture](https://openxla.org/xla/architecture)
**How XLA is organised internally: the front end, the optimizing passes shared across targets, and the per-backend code generation.**

Read this if you want the stage names behind what we observe in the chapter. Fusion and layout assignment are passes in the sequence described here.

---

## Reference 3: [XLA Operation Semantics](https://openxla.org/xla/operation_semantics)
**The precise definition of every HLO operation: shapes, broadcasting rules and exact numerical behaviour.**

This is a reference, not a read-through. When an HLO dump contains an operation you do not recognise, this is where it is defined.

---

## Reference 4: [Ahead-of-time compilation](https://docs.jax.dev/en/latest/aot.html)
**The staged API we use to reach the compiled program: `.lower()` produces the input to the compiler, `.compile()` runs it.**

Same page as in Chapter 2, consulted here for a different reason: in this chapter we care about what comes *out* of each stage.

---

## Reference 5: [GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
**Practical notes on what makes compiled code fast on a GPU, including the flags that control fusion behaviour.**

We run everything on CPU in this course, but this page is where the ideas from this chapter become actionable on real hardware.

---

## Reference 6: [Profiling JAX programs](https://docs.jax.dev/en/latest/profiling.html)
**How to record and read a trace of what actually executed, with the device timeline.**

The natural next step after this chapter: we look at what the compiler produced, profiling shows you what it then cost at run time.

---
