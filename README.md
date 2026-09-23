# Part I: Preparation for Finite Element Method (FEM): FEniCS

This guide shows how to properly install **FEniCS** (the classic `dolfin` version) in a Conda environment and use it inside **Jupyter Notebook**.

> **Important**: Do **not** use `pip install fenics`—it often fails on Linux/WSL because FEniCS requires compiled C++ dependencies. Always use **Conda from the `conda-forge` channel**.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed
- Terminal access (Linux, macOS, Anaconda prompt, or WSL on Windows)

## Installation Steps

Please run the following installation commands in the appropriate terminal for your setup:

- **Anaconda Prompt** — if you are using **Jupyter Notebook launched from Anaconda (on Windows/macOS/Linux)**.  
- **Ubuntu/Debian terminal in WSL** — if you are running a Linux distribution (e.g., Ubuntu) via **Windows Subsystem for Linux (WSL)** on Windows.  
- **Native Linux terminal** — if you are working directly on a Linux system (e.g., Ubuntu, Debian, etc.).

> **Why?** FEniCS must be installed in the same Python environment that Jupyter Notebook/Lab uses. Using the correct terminal ensures you’re operating in the right Conda or system context.

---

### 1. Create a dedicated Conda environment

```bash
conda create -n fenics-env -c conda-forge fenics jupyter
```

This installs:
- FEniCS (`dolfin`)
- Jupyter Notebook
- All required system dependencies (PETSc, UFL, FFC, etc.)

### 2. Activate the environment

```bash
conda activate fenics-env
```

### 3. Install necessary packages

```bash
# Install core scientific computing and visualization libraries
# Even though packages like numpy or matplotlib are common, they are NOT automatically included 
# in a fresh Conda environment and must be explicitly installed to avoid ModuleNotFoundError.
conda install -c conda-forge numpy scipy matplotlib

# Install mesh handling tools
# - meshio: reads/writes various mesh formats (e.g., .msh, .xdmf); essential for data exchange between Gmsh and FEniCS
# - gmsh: open-source mesh generator for complex geometries and high-quality finite element meshes
conda install - c conda-forge meshio gmsh

# Install image processing library
# - pillow (PIL): the modern replacement for Python Imaging Library; used to generate, edit, and save images, including GIF animations
conda install -c conda-forge pillow
```

> **Why install numpy/matplotlib again?**  
> Although FEniCS may pull in some dependencies indirectly, the versions might be incomplete or incompatible. Explicit installation ensures stability and prevents runtime errors.

> **Use `conda-forge` consistently**: Mixing `pip` and `conda` for scientific packages often leads to ABI conflicts. Sticking to `conda-forge` guarantees binary compatibility across all libraries.

### 4. Launch Jupyter Notebook

Depending on your setup, launch Jupyter Notebook as follows:

#### **On Windows with Anaconda (via Anaconda Prompt)**
```bash
jupyter notebook
```
- This will **automatically open Jupyter in your default web browser**.
- If it doesn’t, check the terminal output for a URL like `http://localhost:8888/?token=...` and paste it into your browser.

#### **On WSL (Windows Subsystem for Linux) or native Linux**
```bash
jupyter notebook --no-browser --port=8888
```
- **Do not expect a browser window to open**—WSL/Linux has no GUI by default.
- Copy the full URL from the terminal (it contains a security token).
- Paste it into a **web browser on your Windows host**.
  WSL 2 automatically forwards `localhost:8888` to Windows, so `http://localhost:8888` just works!

> **Tip**: You can also use `jupyter lab` instead of `jupyter notebook` if you prefer JupyterLab.

---

## Verify Installation

In a new notebook cell, run:

```python
from dolfin import *
print("FEniCS version:", dolfin.__version__)
print("Success! FEniCS is ready to use.")
```

If no error appears, you're all set!

## Common Mistakes to Avoid

| What **not** to do | What to do instead |
|----------------------|---------------------|
| `pip install fenics` in terminal or notebook | Use `conda install -c conda-forge fenics` |
| Installing FEniCS in the base Conda environment | Use a **dedicated environment** for reproducibility |
| Mixing `pip` and `conda` for scientific packages | Prefer `conda` for FEniCS, NumPy, SciPy, etc. |

## Resources

- [FEniCS Project Official Site](https://fenicsproject.org/)
- [Conda-Forge FEniCS Package](https://anaconda.org/conda-forge/fenics)
- [WSL installation (Microsoft)](https://aka.ms/wslinstall)

---

>  **Tip**: For new projects, consider [FEniCSx](https://fenicsproject.org/news/2022-09-28-fenicsx-0.4.0/) (`dolfinx`), the next-generation version—just replace `fenics` with `fenicsx` in the installation commands. However, this project uses the classic **FEniCS** (`dolfin`) throughout.

---

# Part II — What Is Inside This Repository

While **Part I** (above) documents the environment setup, this second part describes the
**simulation work** itself: a small, self-contained study of how to go from a scalar diffusion
problem to incompressible Navier–Stokes flow past a cylinder, first in 2D, then in 3D, and finally
in a parallel (MPI) configuration-file-driven form.

Everything is built on **classic FEniCS (`dolfin`)** and uses **Gmsh** as the mesh generator, with
**`meshio`** as the bridge between the two.

## Repository Structure

```
FEM_with_fenics/
├── README.md                                          # Part I: environment setup
├── Diffusion equation.ipynb                           # Case 1 — unsteady diffusion
├── 2d_NS_equation.ipynb                               # Case 2 — 2D incompressible NS
├── 3d_NS_equation_Navier_slip_condition.ipynb         # Case 3 — 3D NS + Navier/Nitsche slip
└── NS_3D_MPI_version/                                 # Case 3 ported to a parallel script
    ├── NS_3D_MPI_RUN.py                               # MPI solver, YAML-config driven
    ├── config.yaml                                    # geometry / mesh / physics / solver / output
    └── RunAll.sh                                      # batch parameter-sweep driver
```

The three notebooks follow an identical nine-step layout, so once you have read one you can navigate
all of them:

| Step | Section | What happens |
|------|---------|--------------|
| 1 | Import necessary packages | `dolfin`, `gmsh`, `meshio`, `numpy`, `matplotlib`, `PIL`, `scipy` |
| 2 | Create mesh | Gmsh constructive geometry (box/rectangle **cut** by cylinder/disk), physical groups, distance-based refinement |
| 3 | Define function space | `VectorFunctionSpace` for velocity, `FunctionSpace` for pressure |
| 4 | Set boundary conditions | `SubDomain` markers + `MeshFunction` + tagged measure `ds` |
| 5 | Define trial and test functions | `TrialFunction` / `TestFunction` / previous-step `Function`s |
| 6 | Define weak equation | UFL variational form, Crank–Nicolson in time, SUPG/PSPG stabilization |
| 7 | Computation | Time loop: assemble → apply BCs → `solve()`; store snapshots |
| 8 | Postprocess | Drag/lift coefficients, FFT of the lift signal, Strouhal number |
| 9 | Plot | Snapshots, per-step PNG sequences, animated GIFs, `.pvd`/`.xdmf` for ParaView |

## Case 1 — `Diffusion equation.ipynb`

Unsteady diffusion

$$\frac{\partial u}{\partial t}-\alpha\nabla^2 u = 0$$

on a $3.0 \times 2.0$ rectangle with a circular hole ($r_c=0.2$) offset from the inlet. The weak form
is Crank–Nicolson ($\theta = 0.5$),

```python
F = (inner((u - u_n)/dt, v)*dx
     + alpha*inner(grad(u_mid), grad(v))*dx
     - inner(f, v)*dx
     + cb * inner(grad(u) * normal, v) * ds)
```

where `cb` is an indicator expression that switches the extra boundary (normal-gradient) term on only
in a thin band around the hole. `alpha = 0.01`, `dt = 0.5*mesh.hmin()`, `T = 10`, linear (`P1`) elements,
and a `bicgstab`/`default` Krylov solve at every step. A parabolic-in-$y$ profile is imposed at the left
edge and zero elsewhere, so the field is transported and damped downstream of the obstacle. Each step is
saved as a PNG under `./alpha_{alpha}_T0_{T0}/` and the sequence is stitched into
`alpha_{alpha}_T0_{T0}.gif` at the end.

> **Note.** This notebook discretises a *scalar* field but keeps it in the **first component of a
> `VectorFunctionSpace`** (`Expression(('T0', 0), ...)`), which is why the code looks vector-valued.
> It is a leftover from sharing scaffolding with the NS cases and is harmless — the second component
> simply stays zero.

## Case 2 — `2d_NS_equation.ipynb`

Incompressible Navier–Stokes past a circular cylinder in a $3.0 \times 2.0$ channel
($r_c = 0.2$, centre at $(0.8, 1.0)$):

$$\frac{\partial \mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}
= -\nabla p + \nu\nabla^2\mathbf{u},\qquad \nabla\cdot\mathbf{u}=0$$

* **Elements:** Taylor–Hood `P2` (velocity) / `P1` (pressure).
* **Mesh:** Gmsh with a `Distance` + `Threshold` background field — `mesh_size_fine = 0.05` near the
  cylinder growing to `0.1` in the far field between `R_inner = 0.3` and `R_outer = 0.6`.
* **Boundary tags** (`MeshFunction` + `SubDomain`): `1` left, `2` right, `3` top, `4` bottom, `5` cylinder.
* **BCs:** uniform inflow `u = (v0, 0)` at the left, no-slip on top/bottom/cylinder, pressure `p = 0` at
  the outlet.
* **Time scheme:** Crank–Nicolson plus a **projection / pressure-correction** splitting: a momentum step
  `Fu` for the velocity, then a pressure-Poisson step `Fp`, iterated 5 times per time step with
  `bicgstab`.
* **Stabilization:** SUPG on the momentum equation and PSPG on the pressure equation, with the
  standard streamwise-length parameter

  ```python
  d1 = 1.0/sqrt((1.0/dt)**2 + (u_mag/h)**2)
  d2 = h*u_mag
  ```
* **Post-processing:** force on the body from the pressure traction
  $\mathbf{F}=\int_{\partial\Omega}(-p\mathbf{I}+\nu\nabla\mathbf{u})\cdot\mathbf{n}\,ds$, integrated over
  `ds(5)`, non-dimensionalised as
  $C_D = F_D/(\tfrac12\rho U^2 A)$, $C_L = F_L/(\tfrac12\rho U^2 A)$ with `rho = 1000` and
  $A = 2r_c H$. `CD`/`CL` histories are plotted, and velocity/pressure snapshots are written as PNG
  sequences and animated GIFs under `./nu_{nu}_v0_{v0}_dh_{hmesh}/{u,p}/`.
* `nu = 0` in the default configuration: the run is an **inviscid, SUPG/PSPG-regularised** demonstration,
  not a validated Reynolds-number study.

## Case 3 — `3d_NS_equation_Navier_slip_condition.ipynb`

The same physics on a 3D box with a cylinder spanning the $z$ direction, with two additions:
**Navier slip on the wall, imposed weakly with Nitsche's method**, and a richer post-processing chain.

* **Geometry (non-dimensionalised by the diameter `D = 0.1`):** $L = 21D$, $H = 14D$, $W = 4D$;
  cylinder centre $(5D, 7D)$, radius $D/2$, axis along $z$.
* **Mesh:** Gmsh `addBox` **cut** by `addCylinder`, first-order (`ElementOrder = 1`) tetrahedra, physical
  surface tags `1 left / 2 right / 3 bottom / 4 top / 5 front / 6 back / 7 cylinder`, volume tag `10 fluid`,
  distance-threshold refinement `0.05 → 0.1`. Written to `cylinder_3d.msh` and converted with
  `meshio` before being loaded by `dolfin`.
* **Boundary tags used by the solver** (re-marked geometrically with `SubDomain`): `1 left, 2 right, 3 top,
  4 bottom, 5 front, 6 back, 7 cylinder`; quadrature order is raised with
  `dx = dx(metadata={'quadrature_degree': 4})`.
* **Weakly enforced wall condition.** There is **no** `DirichletBC` on the cylinder; the wall appears only
  in the variational form through the stress tensor
  $\sigma(\mathbf{u},p) = -p\mathbf{I}+\nu(\nabla\mathbf{u}+\nabla\mathbf{u}^T)$:

  ```python
  nitsche_1 = -inner(dot(sigma(u_mid, p_n), n), v)*ds(7)   # consistency
  nitsche_2 =  inner(v, dot(sigma(u_mid, p_n), n))*ds(7)   # symmetry
  nitsche_3 =  beta * inner(u_t, v_t)*ds(7)                # Navier tangential slip penalty
  nitsche_6 =  alpha/h * dot(u_mid, n)*dot(v, n)*ds(7)     # impermeability penalty
  ```

  with tangential projections `u_t = u_mid - dot(u_mid, n)*n`, `v_t = v - dot(v, n)*n`,
  `alpha = 10.0` and slip coefficient `beta`. **`beta = 0` gives a frictionless (fully slipping) wall;
  increasing `beta` moves the wall towards the classical no-slip condition** — that single knob is the
  point of the whole notebook.
* **Inlet perturbation.** A `RandomInflow(UserExpression)` seeds a spanwise-random profile
  (`epsilon = 0.001`, `numpy.random.seed(42)`) as the initial field `u0` to trigger asymmetry; the
  boundary condition itself is the uniform `Constant((v0, 0, 0))`.
* **Post-processing:** $C_D$ and $C_L$ from the full traction over `ds(7)` with $A = D\cdot H$ and
  $\rho = 1$; time-averaged over the last 100 steps; the lift signal is then analysed with
  `scipy.signal.find_peaks` **and an FFT** to extract the dominant vortex-shedding frequency $f$ and the
  Strouhal number $St = fD/U$. Velocity is exported to `velocity_output.pvd` for ParaView.

## Case 4: The Parallel Version — `NS_3D_MPI_RUN.py`

`NS_3D_MPI_RUN.py` is the 3D case converted into a plain script that runs under MPI with domain
decomposition handled transparently by `dolfin`/PETSc, and with all hard-coded numbers replaced by a
YAML configuration.

**Key differences from the notebook:**

| Aspect | Notebook | Script |
|--------|----------|------------|
| Parameters | hard-coded in cells | `config.yaml` (+ built-in defaults) |
| I/O | every process writes | mesh generated and converted by **rank 0 only**, then `comm.bcast` of paths |
| Wall condition | Nitsche/Navier slip on `ds(7)` | **no-slip** `DirichletBC` on the cylinder |
| Errors | raise locally | `sys.excepthook` prints the rank, then `comm.Abort(1)` for all ranks |
| Output | PNG / GIF / `.pvd` | time-stepped **`.xdmf`** with `flush_output = True`, plus a `run_log.csv` index |
| Progress | text bar | rolling ETA computed on rank 0 from the last 3/4/11 step timings |

**Configuration (`config.yaml`)** — grouped as `geometry`, `mesh`, `physics`, `solver`, `output`. All
lengths are expressed as multiples of the diameter `D`, e.g. `L_factor: 21.0` ⇒ `L = 21 * D`.
The script merges whatever keys you supply over its internal defaults, so a partial file is fine.

**Running:**

```bash
conda activate fenics-env
cd NS_3D_MPI_version

python -u NS_3D_MPI_RUN.py --config config.yaml        # single process
mpiexec -n 4 python -u NS_3D_MPI_RUN.py --config config.yaml   # 4 ranks
```

`--config` defaults to `config.yaml` in the current directory, and the argument parser uses
`parse_known_args()` so the extra flags injected by `mpiexec`/Hydra are ignored instead of crashing.

**Batch sweeps (`RunAll.sh`).** The shell script loops over parameter lists, writes a
`temp_config.yaml` per combination, and pipes each run to a per-parameter log file:

```bash
bash RunAll.sh
```

Set the arrays at the top of the file (`MESH_SIZE_FINE_LIST`, `RC_LIST`, `XC_LIST`, `L_LIST`,
`H_LIST`, `W_LIST`, `v0_LIST`, …) to define the sweep; the nested loops give the full Cartesian
product. Commented-out `mpiexec -n 4/8/12` lines are provided for scaling studies.

## Recommended Reading Order

1. `Diffusion equation.ipynb` — one field, one linear solve per step. Learn the Gmsh → `meshio` → `dolfin`
   pipeline and the `SubDomain`/`MeshFunction`/tagged-`ds` idiom.
2. `2d_NS_equation.ipynb` — adds the pressure/velocity pair, the splitting scheme, SUPG/PSPG and force
   coefficients.
3. `3d_NS_equation_Navier_slip_condition.ipynb` — adds the third direction, Nitsche boundary terms and
   vortex-shedding/Strouhal analysis.
4. `NS_3D_MPI_version/` — the same problem as an industrial-style, reproducible, parallel run.


## Quick Start (TL;DR)

```bash
# 1. environment (Part I above)
conda create -n fenics-env -c conda-forge fenics jupyter
conda activate fenics-env
conda install -c conda-forge numpy scipy matplotlib meshio gmsh pillow

# 2. make the kernel visible in Jupyter (kernel name used by all notebooks: "fenics")
python -m ipykernel install --user --name fenics --display-name "Python (FEniCS)"

# 3. notebooks
jupyter lab          # open Diffusion equation.ipynb first

# 4. parallel case
cd NS_3D_MPI_version
mpiexec -n 4 python -u NS_3D_MPI_RUN.py --config config.yaml
```


## Resources

- [FEniCS Book — Navier–Stokes](https://fenicsproject.org/pub/book.html)
- [SUPG / PSPG stabilisation for incompressible flow](https://en.wikipedia.org/wiki/Streamline_upwind_Petrov_Galerkin_method)
- [Nitsche's method for boundary conditions](https://en.wikipedia.org/wiki/Nitsche%27s_method)
- [Gmsh Python API tutorials](https://gmsh.info/doc/texinfo/gmsh.html#Python-API)
- [meshio](https://github.com/nschloe/meshio)
- [ParaView](https://www.paraview.org/) — open the `*.pvd` / `*.xdmf` outputs for 3D visualisation
