# Introduction to JAX: JIT, Autodiff and the Compiler Stack

Public repository for the *Introduction to JAX: JIT, Autodiff and the Compiler Stack* course.

**Instructor:** Pietropaolo Frisoni. Computational Scientist, Quantum Software Engineer.

---

## How to use this repository

Thank you for visiting this repository!

Each folder corresponds to one chapter of the course. Inside each folder you will find the
Jupyter notebook for that chapter (the same one written on screen during the lectures) and a
`References.md` file with links to the official JAX documentation and further reading, each with
a note on why it is worth opening.

You can follow the folders in the order listed below, which is the same order used in the lectures.
Each chapter assumes the one before it, so I would recommend following that order rather than
jumping around.

I hope you find it useful, and I wish you a lot of fun!

---

## Course order

| # | Folder | Chapter |
|---|--------|---------|
| 1  | `ch00_what_is_jax`                   | Chapter 0: What is JAX and Why Does It Matter *(slides only, no notebook)* |
| 2  | `ch01_jax_internals_execution_model` | Chapter 1: JAX Internals and the Execution Model |
| 3  | `ch02_jit_compilation`               | Chapter 2: Just-In-Time Compilation |
| 4  | `ch03_xla_backend`                   | Chapter 3: The XLA Backend |
| 5  | `ch04_stablehlo_and_export`          | Chapter 4: StableHLO and Model Export |
| 6  | `ch05_primitives`                    | Chapter 5: Primitives and Custom Operations |
| 7  | `ch06_automatic_differentiation`     | Chapter 6: Automatic Differentiation |
| 8  | `ch07_vectorization_with_vmap`       | Chapter 7: Vectorization with vmap |
| 9  | `ch08_parallel_computation`          | Chapter 8: Parallel Computation |
| 10 | `ch09_numerical_computing`           | Chapter 9: Numerical Computing with JAX |
| 11 | `ch10_neural_networks`               | Chapter 10: Neural Networks with JAX |

---

## A note on versions: please read this one

This matters more for JAX than for most libraries.

Every notebook here was recorded and verified against the **exact library versions pinned in
[`requirements.txt`](requirements.txt)**: JAX 0.10.0, jaxlib 0.10.0, Flax 0.12.7, Optax 0.2.8,
NumPy 2.4.4.

JAX releases often, and it makes breaking changes when it does. Functions get renamed, APIs get
deprecated and then removed, and an idiom that is correct today can stop working in a few months.
That is not carelessness. It is the cost of a young compiler stack that is still moving quickly,
and that speed is part of why JAX is worth learning.

So here is an honest commitment, and its limit:

- **The notebooks in this repository will be kept up to date.** When something breaks, this is
  where the fix will appear.
- **The recorded video lectures will not be re-recorded on every release.** Each chapter is decks,
  notebooks and quizzes, all interlocking, and redoing that for every version is not feasible for
  one person. Pretending otherwise would be dishonest.

Which gives a simple rule: install the pinned versions and everything on screen matches exactly.
If you prefer a newer JAX, treat this repository as the current truth and the videos as the
explanation.

And the important part: **the concepts do not move.** Tracing, Jaxprs, compilation, the
transformations, immutability, the key-based random numbers: none of that changes with a version
number. That is what the course is really about. A renamed function takes thirty seconds to look
up. Understanding why it exists is the hard part, and it does not expire.

---

## Setup

Everything in this course runs on an **ordinary laptop CPU**. No GPU is required.

### Option 1: local environment (the one used in the videos)

1. Install [Visual Studio Code](https://code.visualstudio.com) and the **Python** and **Jupyter**
   extensions (both published by Microsoft)
2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with all default options,
   then reopen VSCode
3. Create and activate the course environment:
   ```bash
   conda create -n IntroductionToJax python=3.12
   conda activate IntroductionToJax
   ```
4. Install the pinned packages:
   ```bash
   pip install -r requirements.txt
   ```
5. In VSCode: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one that shows
   `IntroductionToJax` in the path
6. Clone this repository and open the chapter folders in VSCode

> **Note:** this is just the setup used in the videos, for consistency. Any other combination
> (PyCharm, `venv`/`virtualenv`, `uv`, etc.) works just as well. Use whatever you are already
> comfortable with. The JAX code itself does not change :)

### Option 2: Google Colab (no setup required)

Open any `.ipynb` file directly on [Google Colab](https://colab.research.google.com) by uploading
it or pointing Colab at this repository.

Colab ships its own JAX version, which will usually be newer than the pinned one. Most of the
course works unchanged. Where it does not, see the note on versions above.

---

## Prerequisites

- **Python**, comfortably: functions, classes, decorators, comprehensions
- **NumPy**: arrays, shapes, broadcasting, indexing

No compiler background is assumed. Machine learning is not the goal of this course, and most of
the material has nothing to do with it, but Chapter 10 does build a neural network, and knowing
roughly what a loss function and a gradient step are will make that chapter easier.

---

## License

The course content (videos, slides, quizzes) is available on the course platform.
I share the notebooks here for enrolled students and, more generally, for anyone who wants to
take a look.

This repository is an independent publication and is not affiliated with, endorsed by, or
sponsored by Google or the JAX project. JAX is open-source software released under the
Apache 2.0 license.
