# Chapter 0: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html)
**The official starting page: installation, the first array operations, and a short tour of the four transformations the whole library is built on.**

Worth skimming even after this chapter, because it shows in a few minutes the shape of everything we take apart slowly over the rest of the course.

---

## Reference 2: [Thinking in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)
**The page that contrasts JAX with NumPy directly: what looks the same, what behaves differently, and why the differences are consequences of the functional model rather than arbitrary choices.**

If you read one official page before starting, make it this one. It is the closest thing in the documentation to the argument this chapter makes.

---

## Reference 3: [Key Concepts](https://docs.jax.dev/en/latest/key-concepts.html)
**Definitions of the vocabulary used everywhere else: arrays, transformations, tracing, jaxprs and pytrees, each explained in a paragraph.**

Useful as a glossary while watching the early chapters. Several of these words are used casually in blog posts as if everyone already knew them.

---

## Reference 4: [JAX Glossary](https://docs.jax.dev/en/latest/glossary.html)
**A compact alphabetical list of JAX terminology, from `abstract value` to `weak type`.**

Short, and surprisingly handy when a term shows up in an error message and you are not sure whether it is a JAX concept or an XLA one.

---

## Reference 5: [Installation Guide](https://docs.jax.dev/en/latest/installation.html)
**Official installation instructions for CPU, GPU and TPU, including the difference between the `jax` and `jaxlib` packages and which versions go together.**

The CPU install is all this course needs. The GPU section is worth bookmarking for later, because the pairing of jax and jaxlib versions is the thing people get wrong most often.

---

## Reference 6: [JAX on GitHub](https://github.com/jax-ml/jax)
**The source repository: issues, discussions, release notes and the changelog.**

The changelog is the fastest way to find out whether something that broke in your code was deprecated on purpose. Given how quickly JAX moves, it is worth knowing where it lives.

---
