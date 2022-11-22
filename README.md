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

Authors : Michel Duprez, Vanessa Lleras, Alexei Lozinski and Killian Vuillemot. 