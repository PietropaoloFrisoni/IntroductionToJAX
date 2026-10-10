# Chapter 6: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [Automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
**The official introduction to `jax.grad`: differentiating scalar functions, gradients with respect to nested structures, and combining differentiation with the other transformations.**

The natural companion to the first half of this chapter, and a good check that the mental model you built from the lecture matches the official one.

---

## Reference 2: [The Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
**A long notebook covering forward and reverse mode, Jacobians, Hessians, Jacobian-vector and vector-Jacobian products, and how to compose them efficiently.**

The single most useful page on this topic anywhere. The section explaining when to use `jacfwd` and when `jacrev` saves real time, and the reasoning generalises beyond JAX.

---

## Reference 3: [jax.grad](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)
**API reference, including `argnums`, `has_aux` and `holomorphic`.**

`has_aux` is the argument worth knowing about: it is how you return a loss and its diagnostics from one differentiated function without computing anything twice.

---

## Reference 4: [jax.value_and_grad](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html)
**Reference for the variant that returns both the value and its gradient in a single pass.**

This is what a training loop actually uses. Calling the function once for the loss and again for the gradient does twice the work for no reason.

---

## Reference 5: [jax.jacfwd](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html)
**Reference for the forward-mode Jacobian, with its complement `jax.jacrev`.**

The choice between the two comes down to the shape of your function: tall favours forward mode, wide favours reverse. The cookbook above explains why.

---

## Reference 6: [jax.hessian](https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html)
**Reference for second derivatives, defined in terms of composing the two Jacobian modes.**

Reading its definition is instructive in itself: it shows that higher-order differentiation is composition, not a separate mechanism.

---

## Reference 7: [jax.lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
**Reference for the structured loop that carries state across iterations and remains differentiable.**

The one to understand properly. A Python loop is unrolled at trace time and compiles slowly, while `scan` stays a single operation no matter how many steps it runs.

---

## Reference 8: [Control flow and logical operators](https://docs.jax.dev/en/latest/control-flow.html)
**The structured alternatives to Python control flow (`cond`, `while_loop`, `fori_loop`, `scan`) and which of them are differentiable.**

Worth re-reading at this point in the course. The table of what supports reverse-mode differentiation answers a question that comes up constantly.

---
