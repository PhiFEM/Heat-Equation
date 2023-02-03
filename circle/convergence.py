import dolfin as df
import numpy as np
import sympy
import time
import matplotlib.pyplot as plt
import mshr
import os

plt.style.use("ggplot")

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["krylov_solver"]["nonzero_initial_guess"] = True

# test case :
# 1) L2H1 error , P^1
# 2) LinfL2 error, P^1
# 3) L2H1 error , P^2
# 4) LinfL2 error, P^2
test_case = 1

if test_case == 1:
    # expected slope 1
    # power on dt
    exp_dt = 1.0
    # Number of iterations
    Iter = 5
    degV = 1
elif test_case == 2:
    # expected slope 2
    # power on dt
    exp_dt = 2.0
    # Number of iterations
    Iter = 5
    degV = 1
elif test_case == 3:
    # expected slope 2
    # power on dt
    exp_dt = 2.0
    # Number of iterations
    Iter = 5
    degV = 2
elif test_case == 4:
    # expected slope 3
    # power on dt
    exp_dt = 3.0
    # Number of iterations
    Iter = 4
    degV = 2

# Final time of the simulation
T = 1.0
# parameter of the ghost penalty
sigma = 1.0
# Polynome Pk
degPhi = degV + 1
# Ghost penalty
ghost = True
# compare the computation times or not
compute_times = True
# save results
write_output = True


# Function used to write in the outputs files
def output_latex(f, A, B):
    for i in range(len(A)):
        f.write("(")
        f.write(str(A[i]))
        f.write(",")
        f.write(str(B[i]))
        f.write(")\n")
    f.write("\n")


if not os.path.exists("./outputs"):
    os.makedirs("./outputs")

f = open(
    f"./outputs/output_euler_homo_circle_manufactured_l_{degPhi}_T_{T}_dt_{int(exp_dt)}_h_P{degV}_sigma_{sigma}.txt",
    "w",
)

# Computation of the exact solution and exact source term
t, x, y = sympy.symbols("tt xx yy")
u1 = (
    sympy.cos(sympy.pi / 2.0 * (x**2 + y**2)) * sympy.sin(t) * sympy.exp(x)
)
f1 = (
    sympy.diff(u1, t)
    - sympy.diff(sympy.diff(u1, x), x)
    - sympy.diff(sympy.diff(u1, y), y)
)

#######################################
#              Phi-FEM                #
#######################################

# Initialistion of the output
size_mesh_phi_fem_vec = np.zeros(Iter)
error_L2_phi_fem_vec = np.zeros(Iter)
error_H1_phi_fem_vec = np.zeros(Iter)
size_matrices_phi_fem = np.zeros(Iter)
if compute_times:
    computation_time_phi_fem = np.zeros(Iter)
    assembly_time_phi_fem = np.zeros(Iter)
    solve_time_phi_fem = np.zeros(Iter)


for i in range(0, Iter):
    print("###########################")
    text = " Iteration phi FEM " + str(i + 1) + " "
    print(f"{text:#^27}")
    print("###########################")

    # Construction of the mesh
    N = int(8 * 2 ** ((i)))
    mesh_macro = df.RectangleMesh(
        df.Point(-1.5, -1.5), df.Point(1.5, 1.5), N, N
    )
    dt = mesh_macro.hmax() ** exp_dt
    Time = np.arange(0, T, dt)
    V_phi = df.FunctionSpace(mesh_macro, "CG", degPhi)
    phi = df.Expression(
        "-(1.) + ((x[0])*(x[0]) + (x[1])*(x[1]))",
        degree=degPhi,
        domain=mesh_macro,
    )
    phi = df.interpolate(phi, V_phi)
    domains = df.MeshFunction(
        "size_t", mesh_macro, mesh_macro.topology().dim()
    )
    domains.set_all(0)
    for ind in range(mesh_macro.num_cells()):
        mycell = df.Cell(mesh_macro, ind)
        v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
        if phi(v1x, v1y) <= 0 or phi(v2x, v2y) <= 0 or phi(v3x, v3y) <= 0:
            domains[ind] = 1
    mesh = df.SubMesh(mesh_macro, domains, 1)
    V = df.FunctionSpace(mesh, "CG", degV)

    # Construction of phi
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = df.Expression(
        "-(1.) + ((x[0])*(x[0]) + (x[1])*(x[1]))", degree=degPhi, domain=mesh
    )
    phi = df.interpolate(phi, V_phi)

    # Facets and cells where we apply the ghost penalty
    mesh.init(1, 2)
    facet_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    cell_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    facet_ghost.set_all(0)
    cell_ghost.set_all(0)
    for mycell in df.cells(mesh):
        for myfacet in df.facets(mycell):
            v1, v2 = df.vertices(myfacet)
            if (
                phi(v1.point().x(), v1.point().y())
                * phi(v2.point().x(), v2.point().y())
                < 0
            ):
                cell_ghost[mycell] = 1
                for myfacet2 in df.facets(mycell):
                    facet_ghost[myfacet2] = 1

    # Computation of the source term and exact solution
    f_expr = []
    u_expr = []
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
        u_expr += [
            df.Expression(
                sympy.ccode(u1)
                .replace("xx", "x[0]")
                .replace("yy", "x[1]")
                .replace("tt", "temps"),
                temps=temps,
                degree=4,
                domain=mesh,
            )
        ]

    # Initialize cell function for domains
    dx = df.Measure("dx")(domain=mesh, subdomain_data=cell_ghost)
    ds = df.Measure("ds")(domain=mesh)
    dS = df.Measure("dS")(domain=mesh, subdomain_data=facet_ghost)

    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    w = df.TrialFunction(V)
    v = df.TestFunction(V)

    # first order stabilization
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
            phi * w * dt ** (-1) - df.div(df.grad(phi * w)),
            df.div(df.grad(phi * v)),
        ) * dx(1)

    if compute_times:
        start_assemble = time.time()
    A = df.assemble(a)
    if compute_times:
        end_assemble = time.time()
        assembly_time_phi_fem[i] = end_assemble - start_assemble

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

        B = df.assemble(L)
        if compute_times:
            start_solve = time.time()
        df.solve(A, w_n1.vector(), B)
        if compute_times:
            end_solve = time.time()
            solve_time_phi_fem[i] += end_solve - start_solve
            computation_time_phi_fem[i] += end_solve - start_solve
        wn = w_n1
        sol += [phi * wn]
        print("(", i + 1, ",", ind, "/", len(Time) - 1, ")")
    if compute_times:
        computation_time_phi_fem[i] += assembly_time_phi_fem[i]
    size_matrices_phi_fem[i] = np.shape(A.array())[0]

    # Computation of the error
    norm_L2_exact = 0.0
    err_L2 = 0.0
    norm_H1_exact = 0.0
    err_H1 = 0.0
    for j in range(len(Time)):
        norm_L2_exact_j = df.assemble(u_expr[j] ** 2 * dx)
        if norm_L2_exact < norm_L2_exact_j:
            norm_L2_exact = norm_L2_exact_j
        err_L2_j = df.assemble((sol[j] - u_expr[j]) ** 2 * dx)
        if err_L2 < err_L2_j:
            err_L2 = err_L2_j
        norm_H1_exact += df.assemble(dt * df.grad(u_expr[j]) ** 2 * dx)
        err_H1 += df.assemble(dt * df.grad(sol[j] - u_expr[j]) ** 2 * dx)
    err_L2 = err_L2**0.5 / norm_L2_exact**0.5
    err_H1 = err_H1**0.5 / norm_H1_exact**0.5
    error_L2_phi_fem_vec[i] = err_L2
    error_H1_phi_fem_vec[i] = err_H1
    size_mesh_phi_fem_vec[i] = mesh.hmax()
    print("h :", mesh.hmax())
    print("relative L2 error : ", err_L2)
    print("relative H1 error : ", err_H1)
    print("")

if write_output:
    f.write("relative L2 norm phi fem : \n")
    output_latex(f, size_mesh_phi_fem_vec, error_L2_phi_fem_vec)
    f.write("relative H1 norm phi fem : \n")
    output_latex(f, size_mesh_phi_fem_vec, error_H1_phi_fem_vec)
    if compute_times:
        f.write("assembly time phi-fem LinfL2: \n")
        output_latex(f, assembly_time_phi_fem, error_L2_phi_fem_vec)
        f.write("assembly time phi-fem L2H1: \n")
        output_latex(f, assembly_time_phi_fem, error_H1_phi_fem_vec)
        f.write("solve time phi-fem LinfL2: \n")
        output_latex(f, solve_time_phi_fem, error_L2_phi_fem_vec)
        f.write("solve time phi-fem L2H1: \n")
        output_latex(f, solve_time_phi_fem, error_H1_phi_fem_vec)
        f.write("total computation time phi-fem LinfL2: \n")
        output_latex(f, computation_time_phi_fem, error_L2_phi_fem_vec)
        f.write("total computation time phi-fem L2H1: \n")
        output_latex(f, computation_time_phi_fem, error_H1_phi_fem_vec)

#######################################
#           Standard FEM              #
#######################################
size_mesh_standard_vec = np.zeros(Iter)
error_L2_standard_vec = np.zeros(Iter)
error_H1_standard_vec = np.zeros(Iter)
size_matrices_standard_fem = np.zeros(Iter)
if compute_times:
    computation_time_standard_fem = np.zeros(Iter)
    solve_time_standard_fem = np.zeros(Iter)
    assembly_time_standard_fem = np.zeros(Iter)

domain_mesh = mshr.Circle(df.Point(0.0, 0.0), 1.0)  # creation of the domain
for i in range(0, Iter):
    print("###########################")
    text = " Iteration standard " + str(i + 1) + " "
    print(f"{text:#^27}")
    print("###########################")

    # Construction of the mesh
    N = int(8 * 2 ** (i - 1))
    mesh = mshr.generate_mesh(domain_mesh, N)
    dt = mesh.hmax() ** exp_dt
    Time = np.arange(0, T, dt)
    V = df.FunctionSpace(mesh, "CG", degV)

    # Computation of the source term
    f_expr, u_expr = [], []
    for temps in Time:
        f_expr = f_expr + [
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
        u_expr = u_expr + [
            df.Expression(
                sympy.ccode(u1)
                .replace("xx", "x[0]")
                .replace("yy", "x[1]")
                .replace("tt", "temps"),
                temps=temps,
                degree=4,
                domain=mesh,
            )
        ]

    # Initialize cell function for domains
    dx = df.Measure("dx")(domain=mesh)
    ds = df.Measure("ds")(domain=mesh)
    dS = df.Measure("dS")(domain=mesh)
    # Resolution
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    u_n = df.Expression("0.0", degree=degV + 2, domain=mesh)
    sol = [u_n]
    a = dt ** (-1) * u * v * dx + df.inner(df.grad(u), df.grad(v)) * dx
    uD = df.Expression("0.0", degree=degV, domain=mesh)
    bc = df.DirichletBC(V, uD, "on_boundary")
    if compute_times:
        start_assemble = time.time()
    A = df.assemble(a)
    if compute_times:
        end_assemble = time.time()
        assembly_time_standard_fem[i] = end_assemble - start_assemble
    for ind in range(1, len(Time)):
        L = dt ** (-1) * u_n * v * dx + f_expr[ind] * v * dx
        u_n1 = df.Function(V)
        B = df.assemble(L)
        bc.apply(A)
        bc.apply(B)
        if compute_times:
            start_solve = time.time()
        df.solve(A, u_n1.vector(), B)
        if compute_times:
            end_solve = time.time()
            solve_time_standard_fem[i] += end_solve - start_solve
            computation_time_standard_fem[i] += end_solve - start_solve
        sol = sol + [u_n1]
        u_n = u_n1
        print("(", i + 1, ",", ind, "/", len(Time) - 1, ")")
    size_matrices_standard_fem[i] = np.shape(A.array())[0]
    if compute_times:
        computation_time_standard_fem[i] += assembly_time_standard_fem[i]
    # Computation of the error
    norm_L2_exact = 0.0
    err_L2 = 0.0
    norm_H1_exact = 0.0
    err_H1 = 0.0
    for j in range(len(Time)):
        norm_L2_exact_j = df.assemble(u_expr[j] ** 2 * dx)
        if norm_L2_exact < norm_L2_exact_j:
            norm_L2_exact = norm_L2_exact_j
        err_L2_j = df.assemble((sol[j] - u_expr[j]) ** 2 * dx)
        if err_L2 < err_L2_j:
            err_L2 = err_L2_j
        norm_H1_exact += df.assemble(dt * df.grad(u_expr[j]) ** 2 * dx)
        err_H1 += df.assemble(dt * df.grad(sol[j] - u_expr[j]) ** 2 * dx)
    err_L2 = err_L2**0.5 / norm_L2_exact**0.5
    err_H1 = err_H1**0.5 / norm_H1_exact**0.5
    size_mesh_standard_vec[i] = mesh.hmax()
    error_L2_standard_vec[i] = err_L2
    error_H1_standard_vec[i] = err_H1
    print("h :", mesh.hmax())
    print("relative L2 error : ", err_L2)
    print("relative H1 error : ", err_H1)
    print("")


if write_output:
    f.write("relative L2 norm standard fem : \n")
    output_latex(f, size_mesh_standard_vec, error_L2_standard_vec)
    f.write("relative H1 norm standard fem : \n")
    output_latex(f, size_mesh_standard_vec, error_H1_standard_vec)
    if compute_times:
        f.write("assembly time standard fem LinfL2: \n")
        output_latex(f, assembly_time_standard_fem, error_L2_standard_vec)
        f.write("assembly time standard fem L2H1: \n")
        output_latex(f, assembly_time_standard_fem, error_H1_standard_vec)
        f.write("solve time standard fem LinfL2: \n")
        output_latex(f, solve_time_standard_fem, error_L2_standard_vec)
        f.write("solve time standard fem L2H1: \n")
        output_latex(f, solve_time_standard_fem, error_H1_standard_vec)
        f.write("total computation time standard fem LinfL2: \n")
        output_latex(f, computation_time_standard_fem, error_L2_standard_vec)
        f.write("total computation time standard fem L2H1: \n")
        output_latex(f, computation_time_standard_fem, error_H1_standard_vec)

# Print the output vectors phi_fem
print("Vector h phi fem:", size_mesh_phi_fem_vec)
print("Vector relative L2 error phi fem : ", error_L2_phi_fem_vec)
print("Vector relative H1 error phi fem : ", error_H1_phi_fem_vec)

# Print the output vectors standard fem
print("Vector h standard fem:", size_mesh_standard_vec)
print("Vector relative L2 error standard fem : ", error_L2_standard_vec)
print("Vector relative H1 error standard fem : ", error_H1_standard_vec)

if write_output:
    f.close()

plt.figure()
if test_case == 1 or test_case == 3:
    plt.subplot(2, 2, 1)
    plt.loglog(size_mesh_phi_fem_vec, error_H1_phi_fem_vec, label="H1 phiFEM")
    plt.loglog(size_mesh_standard_vec, error_H1_standard_vec, label="H1 std")
    plt.loglog(
        size_mesh_phi_fem_vec,
        [h ** (degV) for h in size_mesh_phi_fem_vec],
        label="O(h^" + str(degV) + ")",
    )
    plt.legend()
    plt.xlabel("h")
    plt.ylabel("error")

    plt.subplot(2, 2, 2)
    plt.loglog(
        computation_time_phi_fem, error_H1_phi_fem_vec, label="H1 phiFEM"
    )
    plt.loglog(
        computation_time_standard_fem, error_H1_standard_vec, label="H1 std"
    )
    plt.legend()
    plt.xlabel("computation time")
    plt.ylabel("error")

    plt.subplot(2, 2, 3)
    plt.loglog(size_matrices_phi_fem, assembly_time_phi_fem, label="phiFEM")
    plt.loglog(
        size_matrices_standard_fem, assembly_time_standard_fem, label="std"
    )
    plt.legend()
    plt.xlabel("size matrices")
    plt.ylabel("assembly time")

    plt.subplot(2, 2, 4)
    plt.loglog(solve_time_phi_fem, error_H1_phi_fem_vec, label="H1 phiFEM")
    plt.loglog(solve_time_standard_fem, error_H1_standard_vec, label="H1 std")
    plt.legend()
    plt.xlabel("solve time")
    plt.ylabel("error")

elif test_case == 2 or test_case == 4:
    plt.subplot(2, 2, 1)
    plt.loglog(size_mesh_phi_fem_vec, error_L2_phi_fem_vec, label="L2 phiFEM")
    plt.loglog(size_mesh_standard_vec, error_L2_standard_vec, label="L2 std")
    plt.loglog(
        size_mesh_phi_fem_vec,
        [h ** (degV + 1) for h in size_mesh_phi_fem_vec],
        label="O(h^" + str(degV + 1) + ")",
    )
    plt.legend()
    plt.xlabel("h")
    plt.ylabel("error")

    plt.subplot(2, 2, 2)
    plt.loglog(
        computation_time_phi_fem, error_L2_phi_fem_vec, label="L2 phiFEM"
    )
    plt.loglog(
        computation_time_standard_fem, error_L2_standard_vec, label="L2 std"
    )
    plt.legend()
    plt.xlabel("computation time")
    plt.ylabel("error")

    plt.subplot(2, 2, 3)
    plt.loglog(size_matrices_phi_fem, assembly_time_phi_fem, label="phiFEM")
    plt.loglog(
        size_matrices_standard_fem, assembly_time_standard_fem, label="std"
    )
    plt.legend()
    plt.xlabel("size matrices")
    plt.ylabel("assembly time")

    plt.subplot(2, 2, 4)
    plt.loglog(solve_time_phi_fem, error_L2_phi_fem_vec, label="L2 phiFEM")
    plt.loglog(solve_time_standard_fem, error_L2_standard_vec, label="L2 std")
    plt.legend()
    plt.xlabel("solve time")
    plt.ylabel("error")

plt.tight_layout()
plt.savefig(
    f"./outputs/plot_problem_euler_test_case_{test_case}_sigma_{sigma}.png"
)
plt.show()
