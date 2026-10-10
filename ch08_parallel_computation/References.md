# Chapter 8: References
### Introduction to JAX: JIT, Autodiff and the Compiler Stack

---

## Reference 1: [Introduction to parallel programming](https://docs.jax.dev/en/latest/sharded-computation.html)
**The modern guide to running across several devices: sharding, meshes, and letting the compiler place the computation.**

This is the API to learn first if you are starting today. `pmap` still works and is still worth understanding, but new code tends to be written this way.

---

## Reference 2: [jax.pmap](https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html)
**Reference for the explicit parallel map, including `axis_name` and the collective operations it enables.**

The explicit model is easier to reason about when you are learning, because you can see exactly where each piece of data goes.

---

## Reference 3: [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
**A notebook showing how an array can live across devices and how a compiled function is partitioned to follow it.**

Good to read right after the chapter: it shows the same ideas with pictures of the sharding, which helps more than it sounds.

---

## Reference 4: [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
**The explicit per-shard programming model: you write the code for one shard and declare how data is split.**

The middle ground between `pmap` and fully automatic sharding, and the one to reach for when the compiler's choices are not what you want.

---

## Reference 5: [jax.make_mesh](https://docs.jax.dev/en/latest/_autosummary/jax.make_mesh.html)
**Reference for constructing the device mesh that sharding specifications refer to.**

Short page, but the mesh and its axis names are the vocabulary everything else in this area is written in.

---

## Reference 6: [jax.device_put](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html)
**Reference for placing an array on a specific device or according to a sharding.**

The simplest way to see sharding in action before any parallel function is involved.

---

## Reference 7: [Multi-process environments](https://docs.jax.dev/en/latest/multi_process.html)
**How JAX programs span several hosts, not just several devices on one machine.**

Beyond the scope of this course, but worth knowing it exists: the step from one machine to a cluster is smaller than you would expect.

---
