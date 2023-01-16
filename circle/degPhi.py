import dolfin as df
import numpy as np
import sympy
import time
import matplotlib.pyplot as plt
import os

plt.style.use("ggplot")

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True

# test case :
# 1) L2H1 error , P^1
# 2) LinfL2 error, P^1
# 3) L2H1 error , P^2
# 4) LinfL2 error, P^2
test_case = 2

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

plt.figure()

# Final time of the simulation
T = 1.0
# parameter of the ghost penalty
sigma = 1.0
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


if not os.path.exists("./outputs"):
    os.makedirs("./outputs")
    
f = open(
    f"./outputs/output_euler_homo_circle_manufactured_degPhi_T_{T}_dt_{int(exp_dt)}_h_P{degV}_sigma_{sigma}.txt",
    "w",
)
# Computation of the exact solution and exact source term
t, x, y = sympy.symbols("tt xx yy")
u1 = sympy.cos(sympy.pi / 2.0 * (x**2 + y**2)) * sympy.sin(t) * sympy.exp(x)
f1 = (
    sympy.diff(u1, t)
    - sympy.diff(sympy.diff(u1, x), x)
    - sympy.diff(sympy.diff(u1, y), y)
)

#######################################
#              Phi-FEM                #
#######################################

for k in range(0, 4):
    degPhi = degV + k
    # Initialistion of the output
    size_mesh_phi_fem_vec = np.zeros(Iter)
    error_L2_phi_fem_vec = np.zeros(Iter)
    error_H1_phi_fem_vec = np.zeros(Iter)
    cond_phi_fem_vec = np.zeros(Iter)
    size_matrices_phi_fem = np.zeros(Iter)
    if compute_times:
        computation_time_phi_fem = np.zeros(Iter)
        assembly_time_phi_fem = np.zeros(Iter)
        solve_time_phi_fem = np.zeros(Iter)

    for i in range(0, Iter):
        print("###########################")
        text = " Iteration phi FEM " + str(i + 1) + " "
        print(f"{text:#^27}")
        text = " l = " + str(degPhi) + " "
        print(f"{text:#^27}")
        print("###########################")

        # Construction of the mesh
        N = int(8 * 2 ** ((i)))
        mesh_macro = df.RectangleMesh(df.Point(-1.5, -1.5), df.Point(1.5, 1.5), N, N)
        dt = mesh_macro.hmax() ** exp_dt
        Time = np.arange(0, T, dt)
        V_phi = df.FunctionSpace(mesh_macro, "CG", degPhi)
        phi = df.Expression(
            "-(1.) + ((x[0])*(x[0]) + (x[1])*(x[1]))", degree=degPhi, domain=mesh_macro
        )
        phi = df.interpolate(phi, V_phi)
        domains = df.MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
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
            if compute_times:
                start_assemble = time.time()
            A = df.assemble(a)
            if compute_times:
                end_assemble = time.time()
                assembly_time_phi_fem[i] += end_assemble - start_assemble
            B = df.assemble(L)
            if compute_times:
                start_solve = time.time()
            df.solve(A, w_n1.vector(), B)
            if compute_times:
                end_solve = time.time()
                solve_time_phi_fem[i] += end_solve - start_solve
                computation_time_phi_fem[i] += (
                    end_assemble - start_assemble + end_solve - start_solve
                )
            wn = w_n1
            sol += [phi * wn]
            print("(", i + 1, ",", ind, "/", len(Time) - 1, ")")
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
        if conditioning == True:
            A = np.matrix(df.assemble(a).array())
            ev, eV = np.linalg.eig(A)
            ev = abs(ev)
            cond = np.max(ev) / np.min(ev)
            cond_phi_fem_vec[i] = cond
            print("conditioning number x h^2", cond)
        print("")

    if write_output:
        f.write(f"{degPhi = } \n")
        f.write("relative L2 norm phi fem : \n")
        output_latex(f, size_mesh_phi_fem_vec, error_L2_phi_fem_vec)
        f.write("relative H1 norm phi fem : \n")
        output_latex(f, size_mesh_phi_fem_vec, error_H1_phi_fem_vec)

    plt.loglog(size_mesh_phi_fem_vec, error_H1_phi_fem_vec, label="l = " + str(degPhi))
plt.legend()
plt.xlabel("h")
if test_case == 1 or test_case == 3 : 
    plt.ylabel("l2(H1) error")
elif test_case == 2 or test_case == 4 : 
    plt.ylabel("linf(L2) error")

plt.tight_layout()
plt.savefig(
    f"./outputs/plot_problem_euler_test_case_{test_case}_sigma_{sigma}_degPhi.png"
)
plt.show()

if write_output:
    f.close()
