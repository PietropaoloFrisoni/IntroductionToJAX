# Chapter 4: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [StableHLO](https://openxla.org/stablehlo)
**The specification of the portable, versioned dialect that JAX lowers to, including its compatibility guarantees.**

The compatibility section is the important one: it explains precisely what "stable" means here, which is the whole reason this layer exists.

---

## Reference 2: [Exporting and serialization](https://docs.jax.dev/en/latest/export/index.html)
**The official guide to `jax.export`: producing a self-contained artifact from a JAX function and loading it elsewhere.**

This is the practical payoff of the chapter. The guarantees about how long an exported artifact stays loadable are stated here and nowhere else.

---

## Reference 3: [jax.export API](https://docs.jax.dev/en/latest/export/export.html)
**Reference for the export module: what goes into an exported object, and the functions that read it back.**

Worth having open while following the chapter, since the vocabulary of the exported object maps directly onto what we inspect.

---

## Reference 4: [jax.export.export](https://docs.jax.dev/en/latest/_autosummary/jax.export.export.html)
**API reference for the entry point itself, with its arguments and return type.**

Short, and the quickest answer to "what exactly am I getting back".

---

## Reference 5: [Shape polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html)
**How to export a function that accepts shapes not known at export time, using symbolic dimensions.**

The part of export most likely to matter in production: without it you are committed to one batch size forever. It takes a second read, and it is worth it.

---

## Reference 6: [Ahead-of-time compilation](https://docs.jax.dev/en/latest/aot.html)
**The staged lowering API, including the cost analysis we query in this chapter.**

The cost analysis is documented here rather than in a page of its own, which is why it is easy to miss.

---
