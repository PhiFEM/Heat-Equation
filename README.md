# $\varphi$-FEM for the heat equation

This repository contains the code used in [Duprez, Michel, and Lleras, Vanessa, and Lozinski, Alexei, and Vuillemot, Killian. *$\varphi$-FEM for the heat equation: optimal convergence on unfitted meshes in space* (preprint)](https://arxiv.org/abs/2303.12013).  

## This repository is for reproducibility purposes only

It is "frozen in time" and not maintained.
To use our latest $\varphi$-FEM code please refer to the [phiFEM repository](https://github.com/PhiFEM/Poisson-Dirichlet-fenicsx).

## Generate the results

The results can be generated using a container image.  
The image is based on [ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30](https://github.com/scientificcomputing/packages/pkgs/container/fenics-gmsh).
It contains also the following python libraries and their dependencies:
- [`dataclasses`](https://docs.python.org/3/library/dataclasses.html),
- [`pygalmesh`](https://pypi.org/project/pygalmesh/),
- [`vedo`](https://vedo.embl.es/).

### Prerequisites

- [Git](https://git-scm.com/),
- [Docker](https://www.docker.com/)/[podman](https://podman.io/).

### Install the image and launch the container

1) Clone this repository in a dedicated directory:
   
   ```bash
   mkdir heat-equation-phifem/
   git clone https://github.com/PhiFEM/publication_Heat-Equation_fenics.git heat-equation-phifem
   ```

2) Download the image from the docker.io registry, in the main directory:
   
   ```bash
   export CONTAINER_ENGINE=docker
   cd heat-equation-phifem
   sudo -E bash pull_image.sh
   ```

3) Launch the container:

   ```bash
   sudo -E bash run_image.sh
   ```

### Example of usage

From the main directory `heat-equation-phifem`, launch the convergence script for the circle test case:

```bash
cd circle/
python3 convergence.py
```

## Python codes 

Implementations of $\varphi$-FEM and standard FEM to solve the heat equation in different cases :

* Folder `circle` : solve the heat equation on a circle, using the manufactured solution $u = \cos\left(\frac{1}{2} \pi (x^2+y^2)\right) \exp(x) \sin(t)$. 
  - `convergence.py` : code to compare the errors of $\varphi$-FEM and a standard FEM on the given problem;
  - `sigma_values.py` : code to emphasize the influence of $\sigma$ on the error;
  - `degree_phi.py` : code to emphasize the influence of $l$, the degree of interpolation of the level-set function, on the error;
  - `plot_meshes.py` : code to plot the domain, an example of mesh used for $\varphi$-FEM and an example of conforming mesh used for standard FEM.


* Folder `popcorn` : solve the heat equation on a popcorn with $f=\exp(-\frac{(x-0.2)^2 + (y-0.3)^2 + (z+0.1)^2}{2\times 0.3^2})$, $u = 0$ on $\Gamma \times (0,T)$ and $u^0 = 0$ in $\Omega$.
  - `main.py` : code to compare the errors of $\varphi$-FEM and a standard FEM on the given problem.
  - `plot_meshes` : code to plot the popcorn domain, an example of mesh used for $\varphi$-FEM and an example of conforming mesh used for standard FEM.

## Issues and support

Please use the issue tracker to report any issues.

## Authors (alphabetical)

[Michel Duprez](https://michelduprez.fr/), Inria Nancy Grand-Est  
[Vanessa Lleras](https://vanessalleras.wixsite.com/lleras), Université de Montpellier  
[Alexei Lozinski](https://orcid.org/0000-0003-0745-0365), Université de Franche-Comté  
[Killian Vuillemot](https://kvuillemot.github.io/), Université de Montpellier
