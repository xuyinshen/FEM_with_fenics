# Preparation for Finite Element Method (FEM): FEniCS + Jupyter Notebook Setup Guide + Other necessary packages

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

## Verify Installation

In a new notebook cell, run:

```python
from dolfin import *
print("FEniCS version:", dolfin.__version__)
print("Success! FEniCS is ready to use.")
```

If no error appears, you're all set!

## Using an Existing Conda Environment?

Already have a Conda environment? Add FEniCS like this:

```bash
conda activate your-env-name
conda install -c conda-forge fenics
conda install ipykernel              # Ensure Jupyter can use this env
python -m ipykernel install --user --name=your-env-name --display-name="Python (FEniCS)"
```

Then select **“Python (FEniCS)”** as the kernel in Jupyter Notebook.

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

> **Tip**: For new projects, consider [FEniCSx](https://fenicsproject.org/news/2022-09-28-fenicsx-0.4.0/) (`dolfinx`), the next-generation version. Replace `fenics` with `fenicsx` in the commands above.
