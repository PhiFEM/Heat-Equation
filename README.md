# $\phi$-FEM for the heat equation

## Python codes 

  Implementations of $\phi$-FEM and standard FEM to solve the Heat equation in different case :
  * Folder `circle` : 
    - `solve_problem.py` : code to solve the heat equation on a circle, using $\phi$-FEM and a standard FEM.
    - `sigma_values.py` : code to solve the heat equation on a circle with $\phi$-FEM, for different values of the parameter $\sigma$.
    - `degree_phi.py` : code to solve the heat equation on a circle with $\phi$-FEM, for different degrees of interpolation of the level-set function $\phi$.
  
  * Folder `star` : 
    - `solve_problem.py` : code to solve the heat equation on a rounded cross, using $\phi$-FEM and a standard FEM.
    - `sigma_values.py` : code to solve the heat equation on a rounded cross with $\phi$-FEM, for different values of the parameter $\sigma$.
    - `degree_phi.py` : code to solve the heaat equation on a rounded cross with $\phi$-FEM, for different degrees of interpolation of the level-set function $\phi$.

Required packages : *FEniCS*, *matplotlib*, *numpy*, *sympy* and *vedo*. 
You can run \[[FEniCS](https://fenicsproject.org/)] using the Docker image with the commands 

```bash
curl -s https://get.fenicsproject.org | bash
fenicsproject run
```

or by installing (on ubuntu) with 

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics
```

Then, just install the other packages with 
```bash
pip install matplotlib 
pip install numpy 
pip install sympy 
pip install vedo

```


Authors : Michel Duprez, Vanessa Lleras, Alexei Lozinski and Killian Vuillemot. 
