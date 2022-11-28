"""
Code to solve the problem on a kind of rounded cross. 
The error is computed with the comparison between the solution of the FEMs and a standard FEM on a very refined mesh.

"""

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import mshr
import time
import sympy


import vedo 
import vedo.dolfin as vdf

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True

# test case :
# 1) delta t = h
# 2) delta t = h**2
test_case = 2

if test_case == 1:
    # puissance on dt
    exp_dt = 1.0
    # Number of iterations
    Iter = 5
    mesh_size = 300

if test_case == 2:
    # puissance on dt
    exp_dt = 2.0
    # Number of iterations
    Iter = 5
    mesh_size = 120

# Time of simulation
T = 1.0
# parameter of the ghost penalty
sigma = 1.0
# Polynome Pk
degV = 1
degPhi = degV + 1
# Ghost penalty
ghost = True
# compare the computation times or not
compute_times = True
# save results
write_output = True
# Compute the conditioning number
conditioning = False

# Function used to write in the outputs files
def output_latex(f, A, B):
    for i in range(len(A)):
        f.write("(")
        f.write(str(A[i]))
        f.write(",")
        f.write(str(B[i]))
        f.write(")\n")
    f.write("\n")


f = open(
    "outputs/output_euler_homo_star_problem_l_{l}_T_{T}_dt_{exp_dt}_h_P{deg}_sigma_{sigma}.txt".format(
        T=int(T), l=degPhi, deg=degV, exp_dt=int(exp_dt), sigma=sigma
    ),
    "w",
)


class phi_expr(df.UserExpression):
    def eval(self, value, x):
        xx, yy = x[0], x[1]
        phi_0 = -1.0 + ((xx) ** 2 + (yy) ** 2)
        phi_1 = -(0.5**2) + ((xx - 0.75) ** 2 + (yy - 0.75) ** 2)
        phi_2 = -(0.5**2) + ((xx - 0.75) ** 2 + (yy + 0.75) ** 2)
        phi_3 = -(0.5**2) + ((xx + 0.75) ** 2 + (yy - 0.75) ** 2)
        phi_4 = -(0.5**2) + ((xx + 0.75) ** 2 + (yy + 0.75) ** 2)
        value[0] = phi_0 * phi_1 * phi_2 * phi_3 * phi_4

    def value_shape(self):
        return (2,)


def omega(xx, yy):
    phi_0 = -1.0 + ((xx) ** 2 + (yy) ** 2)
    phi_1 = -(0.5**2) + ((xx - 0.75) ** 2 + (yy - 0.75) ** 2)
    phi_2 = -(0.5**2) + ((xx - 0.75) ** 2 + (yy + 0.75) ** 2)
    phi_3 = -(0.5**2) + ((xx + 0.75) ** 2 + (yy - 0.75) ** 2)
    phi_4 = -(0.5**2) + ((xx + 0.75) ** 2 + (yy + 0.75) ** 2)

    return phi_0 <= 0.0 and phi_1 >= 0.0 and phi_2 >= 0.0 and phi_3 >= 0.0 and phi_4 >= 0.0


#######################################
#              Exact FEM              #
#######################################
domain = (
    mshr.Circle(df.Point(0, 0), 1)
    - mshr.Circle(df.Point(0.75, 0.75), 0.5)
    - mshr.Circle(df.Point(-0.75, 0.75), 0.5)
    - mshr.Circle(df.Point(0.75, -0.75), 0.5)
    - mshr.Circle(df.Point(-0.75, -0.75), 0.5)
)
mesh_exact = mshr.generate_mesh(domain, mesh_size)
print(
    "###########################\n"
    f"Mesh built using the following parameters : \n    Size of cell on the initial domain :{mesh_exact.hmax()}\n"
    "###########################"
)
V_exact = df.FunctionSpace(mesh_exact, "CG", degV)

# Computation of the source term
t, x, y = sympy.symbols("tt xx yy")
f1 = sympy.cos(sympy.pi*x) * sympy.exp(y) * sympy.sin(t) * sympy.pi
# Define boundary condition
bc_exact = df.DirichletBC(V_exact, df.Constant(0.0), "on_boundary")
dx_exact = df.Measure("dx")(domain=mesh_exact)
dt_exact = mesh_exact.hmax() ** exp_dt
Time_exact = np.arange(0, T, dt_exact)

f_exact = []
for temps in Time_exact:
    f_exact += [
        df.Expression(
            sympy.ccode(f1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("tt", "temps"),
            temps=temps,
            degree=degV + 1,
            domain=mesh_exact,
        )
    ]

# Resolution
u_exact = df.TrialFunction(V_exact)
v_exact = df.TestFunction(V_exact)
u_n_exact = df.Expression("0.0", degree=degV, domain=mesh_exact)
sol_exact = [u_n_exact]
a_exact = (
    dt_exact ** (-1) * u_exact * v_exact * dx_exact
    + df.inner(df.grad(u_exact), df.grad(v_exact)) * dx_exact
)
for ind in range(1, len(Time_exact)):
    L_exact = (
        dt_exact ** (-1) * u_n_exact * v_exact * dx_exact
        + f_exact[ind] * v_exact * dx_exact
    )
    u_n1_exact = df.Function(V_exact)
    df.solve(
        a_exact == L_exact,
        u_n1_exact,
        bc_exact,
        solver_parameters={"linear_solver": "mumps"},
    )
    sol_exact = sol_exact + [u_n1_exact]
    u_n_exact = u_n1_exact
    print("(", ind, "/", len(Time_exact) - 1, ")")
print("Exact solution computed")

#######################################
#              Phi-FEM                #
#######################################

# Initialistion of the output
size_mesh_phi_fem_vec = np.zeros(Iter)
error_L2_phi_fem_vec = np.zeros(Iter)
error_H1_phi_fem_vec = np.zeros(Iter)
cond_phi_fem_vec = np.zeros(Iter)
computation_time_phi_fem = np.zeros(Iter)

size = 1
for i in range(0, Iter):
    print("###########################")
    print("## Iteration phi FEM ", i + 1, "##")
    print("###########################")

    dt = dt_exact * (2 ** (Iter - i))
    print(f"{dt_exact=}")
    print(f"{dt=}")
    Time = np.arange(0, T, dt)

    # Construction of the mesh
    if test_case == 1:
        hmax = 1.e4
        while hmax > dt + df.DOLFIN_EPS:

            mesh_macro = df.RectangleMesh(df.Point(-2, -2), df.Point(2, 2), size, size)
            size += 1
            hmax = mesh_macro.hmax()
        print(f"{hmax=}    {size=}     {dt=}")

    elif test_case == 2:
        hmax = 1.e4
        while hmax > df.sqrt(dt + df.DOLFIN_EPS):

            mesh_macro = df.RectangleMesh(df.Point(-2, -2), df.Point(2, 2), size, size)
            size += 1
            hmax = mesh_macro.hmax()

        print(f"{hmax**2=}    {size=}     {dt=}")

    V_phi = df.FunctionSpace(mesh_macro, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    domains = df.MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
    domains.set_all(0)
    for ind in range(mesh_macro.num_cells()):
        mycell = df.Cell(mesh_macro, ind)
        v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
        if omega(v1x, v1y) or omega(v2x, v2y) or omega(v3x, v3y):
            domains[ind] = 1
    mesh = df.SubMesh(mesh_macro, domains, 1)
    print("num cell mesh:", mesh.num_cells())
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    print(
        "###########################\n"
        f"Mesh built using the following parameters : \n  "
        f"    Size of cells on the background mesh : {mesh_macro.hmax()}\n"
        "###########################"
    )

    V = df.FunctionSpace(mesh, "CG", degV)
    # Computation of the source term and exact solution
    f_expr = []
    for temps in Time:
        f_expr += [
            df.Expression(
                sympy.ccode(f1)
                .replace("xx", "x[0]")
                .replace("yy", "x[1]")
                .replace("tt", "temps"),
                temps=temps,
                degree=degV + 1,
                domain=mesh,
            )
        ]
    # Facets and cells where we apply the ghost penalty
    mesh.init(1, 2)
    facet_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    cell_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    facet_ghost.set_all(0)
    cell_ghost.set_all(0)
    for mycell in df.cells(mesh):
        for myfacet in df.facets(mycell):
            v1, v2 = df.vertices(myfacet)
            if phi(v1.point().x(), v1.point().y()) * phi(
                v2.point().x(), v2.point().y()
            ) < 0 or df.near(
                phi(v1.point().x(), v1.point().y())
                * phi(v2.point().x(), v2.point().y()),
                0.0,
            ):
                cell_ghost[mycell] = 1
                for myfacet2 in df.facets(mycell):
                    facet_ghost[myfacet2] = 1
    # Initialize cell function for domains
    dx = df.Measure("dx")(domain=mesh, subdomain_data=cell_ghost)
    ds = df.Measure("ds")(domain=mesh)
    dS = df.Measure("dS")(domain=mesh, subdomain_data=facet_ghost)

    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    w = df.TrialFunction(V)
    v = df.TestFunction(V)

    def G_h(w, v):
        return (
            sigma
            * df.avg(h)
            * df.dot(df.jump(df.grad(w), n), df.jump(df.grad(v), n))
            * dS(1)
        )

    # Computation of u_0
    wn = df.Expression("0.0", degree=degV, domain=mesh)
    sol = [phi * wn]
    a = (
        dt ** (-1) * phi * w * phi * v * dx
        + df.inner(df.grad(phi * w), df.grad(phi * v)) * dx
        - df.dot(df.inner(df.grad(phi * w), n), phi * v) * ds
    )
    if ghost == True:
        a += G_h(phi * w, phi * v) - sigma * h**2 * df.inner(
            phi * w * dt ** (-1) - df.div(df.grad(phi * w)), df.div(df.grad(phi * v))
        ) * dx(1)

    for ind in range(1, len(Time)):
        L = f_expr[ind] * phi * v * dx + dt ** (-1) * phi * wn * phi * v * dx
        if ghost == True:
            L += (
                -sigma
                * h**2
                * df.inner(f_expr[ind], df.div(df.grad(phi * v)))
                * dx(1)
            )
            L += (
                -sigma
                * h**2
                * df.inner(dt ** (-1) * phi * wn, df.div(df.grad(phi * v)))
                * dx(1)
            )

        w_n1 = df.Function(V)
        A = df.assemble(a)
        B = df.assemble(L)
        start = time.time()
        df.solve(A, w_n1.vector(), B)
        end = time.time()
        computation_time_phi_fem[i] += end - start
        wn = w_n1
        sol += [phi * wn]
        print("(", i + 1, ",", ind, "/", len(Time) - 1, ")")
    # Computation of the error
    norm_L2_exact = 0.0
    err_L2 = 0.0
    norm_H1_exact = 0.0
    err_H1 = 0.0
    j_exact = 0
    for j in range(len(Time)):
        sol_exactj = sol_exact[2 ** (Iter - i) * j]
        Iu_hj = df.project(sol[j], V_exact)
        norm_L2_exact_j = df.assemble(sol_exactj**2 * dx_exact)
        if norm_L2_exact < norm_L2_exact_j:
            norm_L2_exact = norm_L2_exact_j
        err_L2_j = df.assemble((Iu_hj - sol_exactj) ** 2 * dx_exact)
        if err_L2 < err_L2_j:
            err_L2 = err_L2_j
        norm_H1_exact += df.assemble(dt * df.grad(sol_exactj) ** 2 * dx_exact)
        err_H1 += df.assemble(dt * df.grad(Iu_hj - sol_exactj) ** 2 * dx_exact)
        print(f"Time : {j=}   {Time_exact[2**(Iter - i)*j]=}   {Time[j]=}")

    err_L2 = err_L2**0.5 / norm_L2_exact**0.5
    err_H1 = err_H1**0.5 / norm_H1_exact**0.5
    error_L2_phi_fem_vec[i] = err_L2
    error_H1_phi_fem_vec[i] = err_H1
    size_mesh_phi_fem_vec[i] = mesh.hmax()
    print("h :", mesh.hmax())
    print("relative L2 error : ", err_L2)
    print("relative H1 error : ", err_H1)
    if conditioning == True:
        A = np.matrix(df.assemble(a).array())
        ev, eV = np.linalg.eig(A)
        ev = abs(ev)
        cond = np.max(ev) / np.min(ev)
        cond_phi_fem_vec[i] = cond
        print("conditioning number x h^2", cond)
    print("")

if write_output:
    f.write("relative L2 norm phi fem : \n")
    output_latex(f, size_mesh_phi_fem_vec, error_L2_phi_fem_vec)
    f.write("relative H1 norm phi fem : \n")
    output_latex(f, size_mesh_phi_fem_vec, error_H1_phi_fem_vec)
    if compute_times:
        f.write("computation time phi fem LinfL2 : \n")
        output_latex(f, computation_time_phi_fem, error_L2_phi_fem_vec)
        f.write("computation time phi fem L2H1: \n")
        output_latex(f, computation_time_phi_fem, error_H1_phi_fem_vec)


size_mesh_standard_vec = np.zeros(Iter)
error_L2_standard_vec = np.zeros(Iter)
error_H1_standard_vec = np.zeros(Iter)
cond_standard_vec = np.zeros(Iter)
if compute_times:
    computation_time_standard_fem = np.zeros(Iter)

size = 1
for i in range(0, Iter):
    print("###########################")
    print("## Iteration standard ", i + 1, "##")
    print("###########################")

    # Construction of the mesh
    dt = dt_exact * (2 ** (Iter - i))
    print(f"{dt_exact=}")
    print(f"{dt=}")
    Time = np.arange(0, T, dt)
    # Construction of the mesh
    if test_case == 1:
        hmax = 1.e4
        while hmax > dt + df.DOLFIN_EPS:

            mesh = mshr.generate_mesh(domain, size)
            size += 1
            hmax = mesh.hmax()
        print(f"{hmax=}    {size=}     {dt=}")

    elif test_case == 2:
        hmax = 1.e4
        while hmax > df.sqrt(dt + df.DOLFIN_EPS):

            mesh = mshr.generate_mesh(domain, size)
            size += 1
            hmax = mesh.hmax()
        print(f"{hmax**2=}    {size=}     {dt=}")

    V = df.FunctionSpace(mesh, "CG", degV)

    # Computation of the source term
    uD = df.Expression("0.0", degree=degV, domain=mesh)
    f_expr = []
    for temps in Time:
        f_expr += [
            df.Expression(
                sympy.ccode(f1)
                .replace("xx", "x[0]")
                .replace("yy", "x[1]")
                .replace("tt", "temps"),
                temps=temps,
                degree=degV + 1,
                domain=mesh,
            )
        ]

    bc = df.DirichletBC(V, uD, "on_boundary")
    # Initialize cell function for domains
    dx = df.Measure("dx")(domain=mesh)
    ds = df.Measure("ds")(domain=mesh)
    dS = df.Measure("dS")(domain=mesh)
    # Resolution
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    u_n = uD
    sol = [u_n]
    a = dt ** (-1) * u * v * dx + df.inner(df.grad(u), df.grad(v)) * dx
    for ind in range(1, len(Time)):
        L = dt ** (-1) * u_n * v * dx + f_expr[ind] * v * dx
        u_n1 = df.Function(V)
        A = df.assemble(a)
        B = df.assemble(L)
        bc.apply(A)
        bc.apply(B)
        if compute_times:
            start = time.time()
        df.solve(A, u_n1.vector(), B)
        if compute_times:
            end = time.time()
            computation_time_standard_fem[i] += end - start
        sol = sol + [u_n1]
        u_n = u_n1
        print("(", i + 1, ",", ind, "/", len(Time) - 1, ")")

    # Computation of the error
    norm_L2_exact = 0.0
    err_L2 = 0.0
    norm_H1_exact = 0.0
    err_H1 = 0.0
    j_exact = 0
    for j in range(len(Time)):
        sol_exactj = sol_exact[2 ** (Iter - i) * j]
        Iu_hj = df.project(sol[j], V_exact)
        norm_L2_exact_j = df.assemble(sol_exactj**2 * dx_exact)
        if norm_L2_exact < norm_L2_exact_j:
            norm_L2_exact = norm_L2_exact_j
        err_L2_j = df.assemble((Iu_hj - sol_exactj) ** 2 * dx_exact)
        if err_L2 < err_L2_j:
            err_L2 = err_L2_j
        norm_H1_exact += df.assemble(dt * df.grad(sol_exactj) ** 2 * dx_exact)
        err_H1 += df.assemble(dt * df.grad(Iu_hj - sol_exactj) ** 2 * dx_exact)
        print(f"Time : {j=}   {Time_exact[2**(Iter - i)*j]=}   {Time[j]=}")
    err_L2 = err_L2**0.5 / norm_L2_exact**0.5
    err_H1 = err_H1**0.5 / norm_H1_exact**0.5
    size_mesh_standard_vec[i] = mesh.hmax()
    error_L2_standard_vec[i] = err_L2
    error_H1_standard_vec[i] = err_H1
    print("h :", mesh.hmax())
    print("relative L2 error : ", err_L2)
    print("relative H1 error : ", err_H1)
    if conditioning == True:
        A = np.matrix(df.assemble(a).array())
        ev, eV = np.linalg.eig(A)
        ev = abs(ev)
        cond = np.max(ev) / np.min(ev)
        cond_standard_vec[i] = cond
        print("conditioning number x h^2", cond)
    print("")

if write_output:
    f.write("relative L2 norm standard fem : \n")
    output_latex(f, size_mesh_standard_vec, error_L2_standard_vec)
    f.write("relative H1 norm standard fem : \n")
    output_latex(f, size_mesh_standard_vec, error_H1_standard_vec)
    if compute_times:
        f.write("computation time standard fem LinfL2: \n")
        output_latex(f, computation_time_standard_fem, error_L2_standard_vec)
        f.write("computation time standard fem L2H1: \n")
        output_latex(f, computation_time_standard_fem, error_H1_standard_vec)

print("Size cell mesh exact:", mesh_exact.hmax())
# Print the output vectors  phi_fem
print("Vector h phi fem:", size_mesh_phi_fem_vec)
print("Vector relative L2 error phi fem : ", error_L2_phi_fem_vec)
print("Vector relative H1 error phi fem : ", error_H1_phi_fem_vec)
if conditioning and write_output:
    f.write("conditionning number phi fem x h^2 : \n")
    output_latex(f, size_mesh_phi_fem_vec, cond_phi_fem_vec)
    print("conditioning number phi_fem", cond_phi_fem_vec)

# Print the output vectors  standard fem
print("Vector h standard fem:", size_mesh_standard_vec)
print("Vector relative L2 error standard fem : ", error_L2_standard_vec)
print("Vector relative H1 error standard fem : ", error_H1_standard_vec)
if conditioning and write_output:
    f.write("conditionning number standard fem x h^2 : \n")
    output_latex(f, size_mesh_standard_vec, cond_standard_vec)
    print(f"conditioning number standard fem : {cond_standard_vec}")
if write_output:
    f.close()


plt.figure()
if test_case == 1:
    plt.loglog(size_mesh_phi_fem_vec, error_H1_phi_fem_vec, label="H1 phiFEM")
    plt.loglog(size_mesh_standard_vec, error_H1_standard_vec, label="H1 std")
    plt.loglog(size_mesh_phi_fem_vec, size_mesh_phi_fem_vec)
elif test_case == 2:
    plt.loglog(size_mesh_phi_fem_vec, error_L2_phi_fem_vec, label="L2 phiFEM")
    plt.loglog(size_mesh_standard_vec, error_L2_standard_vec, label="L2 std")
    plt.loglog(size_mesh_phi_fem_vec, [h**2 for h in size_mesh_phi_fem_vec])
plt.legend()
plt.savefig(f"outputs/plot_problem_euler_test_case_{test_case}_sigma_{sigma}.png")
plt.show()