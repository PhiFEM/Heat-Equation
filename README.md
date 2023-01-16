# $\phi$-FEM for the heat equation

## Python codes 

  Implementations of $\phi$-FEM and standard FEM to solve the Heat equation in different case :
  * Folder `circle` : solve the heat equation on a circle, using the manufactured solution $u = \cos\left(\frac{1}{2} \pi (x^2+y^2)\right) \exp(x) \sin(t)$. 
    - `convergence.py` : code to compare the errors of $\phi$-FEM and a standard FEM on the given problem;
    - `sigma_values.py` : code to emphasize the influence of $\sigma$ on the error;
    - `degree_phi.py` : code to emphasize the influence of $l$, the degree of interpolation of the level-set function, on the error;
    - `plot_meshes.py` : code to plot the domain, an example of mesh used for $\phi$-FEM and an example of conforming mesh used for standard FEM.
  * Folder `popcorn` : solve the heat equation on a popcorn with $f=\cos(\pi * x) \exp(y) \cos(z)$, $u = 0$ on $\Gamma \times (0,T)$ and $u^0 = 0$ in $\Omega$.
    - `main.py` : code to compare the errors of $\phi$-FEM and a standard FEM on the given problem.
    - `plot_meshes` : code to plot the popcorn domain, an example of mesh used for $\phi$-FEM and an example of conforming mesh used for standard FEM.

Required packages : *FEniCS*, *matplotlib*, *numpy*, *sympy*, *vedo* and *pygalmesh*. 
You can run \[[FEniCS](https://fenicsproject.org/)] using a Docker container with the command

```bash
docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current
```

or by installing (on ubuntu) with 

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics
```

Then, just install the other packages (and dependencies) with 
```bash
pip install matplotlib 
pip install numpy 
pip install sympy 
pip install vedo
sudo apt install libcgal-dev libeigen3-dev
pip install pygalmesh
```


Authors : Michel Duprez, Vanessa Lleras, Alexei Lozinski and Killian Vuillemot. 
