from __future__ import print_function
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import sympy
import time

df.parameters["allow_extrapolation"] = True
import mshr


# Time of simulation
T = 5.0

# Number of iterations
init_Iter = 0
Iter = 4

# parameter of the ghost penalty
sigma = 20.0

# Polynome Pk
polV = 1
degPhi = 2 + polV

# Ghost penalty
ghost = True

# plot the solution
Plot = False

# Compute the conditioning number
conditioning = False

# Parameter for theta-scheme
theta = 1.0

write_output = True
save_figs = False
compute_error = True

# Function used to write in the outputs files
def output_latex(f, A, B):
    for i in range(len(A)):
        f.write("(")
        f.write(str(A[i]))
        f.write(",")
        f.write(str(B[i]))
        f.write(")\n")
    f.write("\n")


if write_output:
    file = open("outputs/{theta}/h/test_case_3_P_{deg}.txt".format(theta = theta, deg=polV), "w")
    file.write("theta = " + str(theta) + "\n")
# Initialistion of the output
size_mesh_phi_fem_vec = np.zeros(Iter)
error_L2_phi_fem_vec = np.zeros(Iter)
error_H1_phi_fem_vec = np.zeros(Iter)
cond_phi_fem_vec = np.zeros(Iter)

domain_mesh = mshr.Polygon(
    [
        df.Point(
            (2 * df.pi**2) / (df.pi**2 + 1), (df.pi**3 - df.pi) / (df.pi**2 + 1)
        ),
        df.Point(0, df.pi),
        df.Point(
            -(2 * df.pi**2) / (df.pi**2 + 1),
            -(df.pi**3 - df.pi) / (df.pi**2 + 1),
        ),
        df.Point(0, -df.pi),
    ]
)
N = int(8 * 2 ** (5))
mesh_standard = mshr.generate_mesh(domain_mesh, N)

if Plot:
    plt.figure()
    df.plot(mesh_standard, color="purple")
    if save_figs:
        plt.savefig("outputs/images/mesh_standard.png")
    plt.show()

for i in range(init_Iter, Iter):

    print("###########################")
    print("## Iteration standard ", i + 1, "##")
    print("###########################")

    # Construction of the mesh
    N = int(10 * 2 ** ((i)))
    mesh_macro = df.RectangleMesh(
        df.Point(-4.0, -4.0), df.Point(4.0, 4.0), 4 * N, 8 * N
    )
    dt = mesh_macro.hmax()  # 10.0*mesh_macro.hmax()**2
    time = np.arange(0, T + dt, dt)

    V_std = df.FunctionSpace(mesh_standard, "CG", polV)

    # Construction of phi
    V_phi = df.FunctionSpace(mesh_standard, "CG", degPhi)
    phi = df.Expression(
        "-(x[1]+x[0]/pi-pi)*(x[1]+x[0]/pi+pi)*(x[1]-pi*x[0]-pi)*(x[1]-pi*x[0]+pi)",
        degree=degPhi,
        domain=mesh_standard,
    )

    phi = df.interpolate(phi, V_phi)

    # Initialize cell function for domains
    dx = df.Measure("dx")(domain=mesh_standard)
    ds = df.Measure("ds")(domain=mesh_standard)
    dS = df.Measure("dS")(domain=mesh_standard)

    # Resolution
    n = df.FacetNormal(mesh_standard)
    h = df.CellDiameter(mesh_standard)
    u = df.TrialFunction(V_std)
    v = df.TestFunction(V_std)
    u_n_std = df.Expression(
        "2.0*sin(x[0])*exp(x[1])",
        degree=polV + 2,
        domain=mesh_standard,
    )  # initial condition

    uD = df.Expression("sin(x[0])*cos(x[1])", degree=polV + 2, domain=mesh_standard)
    sol_std = [u_n_std]
    f = df.Constant("1.0")
    a = dt ** (-1) * u * v * dx + theta * df.inner(df.grad(u), df.grad(v)) * dx
    for ind in range(1, len(time)):

        bc = df.DirichletBC(V_std, uD, "on_boundary")
        L = (
            dt ** (-1) * u_n_std * v * dx
            + theta * f * v * dx
            + (1.0 - theta) * f * v * dx
            - (1.0 - theta) * df.inner(df.grad(u_n_std), df.grad(v)) * dx
        )
        u_n1_std = df.Function(V_std)
        df.solve(a == L, u_n1_std, bc, solver_parameters={"linear_solver": "mumps"})
        sol_std = sol_std + [u_n1_std]
        u_n_std = u_n1_std
        print("(", i + 1, ",", ind, "/", len(time) - 1, ")")

    projected_sol_std = [df.project(sol_std[j], V_std) for j in range(len(time))]

    if Plot:
        plt.close()

        fig = plt.figure()
        for j in range(len(time)):
            df.plot(projected_sol_std[j], title=f"Time = {time[j]}")
            if save_figs:
                plt.savefig(f"outputs/images/plot_standard_{j}.png")
            plt.pause(0.1)
            fig.clear()

    print("###########################")
    print("## Iteration phi FEM ", i + 1, "##")
    print("###########################")

    V_phi = df.FunctionSpace(mesh_macro, "CG", degPhi)
    phi = df.Expression(
        "-(x[1]+x[0]/pi-pi)*(x[1]+x[0]/pi+pi)*(x[1]-pi*x[0]-pi)*(x[1]-pi*x[0]+pi)",
        degree=degPhi,
        domain=mesh_macro,
    )

    phi = df.interpolate(phi, V_phi)
    domains = df.MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())

    def Omega(x, y):
        return (
            y < -x / df.pi + df.pi
            and y < df.pi * x + df.pi
            and y > -x / df.pi - df.pi
            and y > df.pi * x - df.pi
        )

    domains.set_all(0)
    for ind in range(mesh_macro.num_cells()):
        mycell = df.Cell(mesh_macro, ind)
        v1x, v1y, v2x, v2y, v3x, v3y = mycell.get_vertex_coordinates()
        if Omega(v1x, v1y) or Omega(v2x, v2y) or Omega(v3x, v3y):
            domains[ind] = 1
    mesh = df.SubMesh(mesh_macro, domains, 1)
    print(mesh.hmax())
    if save_figs:
        plt.figure()
        df.plot(mesh, color="purple")
        plt.savefig(f"outputs/images/mesh_phi_fem_{i}.png")

    V = df.FunctionSpace(mesh, "CG", polV)

    # Construction of phi
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = df.Expression(
        "-(x[1]+x[0]/pi-pi)*(x[1]+x[0]/pi+pi)*(x[1]-pi*x[0]-pi)*(x[1]-pi*x[0]+pi)",
        degree=degPhi,
        domain=mesh,
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
    u_n = df.Expression(
        "2.0*sin(x[0])*exp(x[1])",
        degree=polV + 2,
        domain=mesh,
    )  # initial condition

    uD = df.Expression("sin(x[0])*cos(x[1])", degree=polV + 2, domain=mesh)
    sol = [u_n]
    f = df.Constant("1.0")
    a = (
        dt ** (-1) * phi * w * phi * v * dx
        + theta * df.inner(df.grad(phi * w), df.grad(phi * v)) * dx
        - theta * df.dot(df.inner(df.grad(phi * w), n), phi * v) * ds
    )

    if ghost == True:
        a += theta * G_h(phi * w, phi * v) - sigma * h**2 * df.inner(
            phi * w * dt ** (-1) - theta * df.div(df.grad(phi * w)),
            df.div(df.grad(phi * v)),
        ) * dx(1)

    for ind in range(1, len(time)):

        L = (
            (theta * f + (1.0 - theta) * f) * phi * v * dx
            + dt ** (-1) * (u_n) * phi * v * dx
            - dt ** (-1) * uD * phi * v * dx
        )
        L += (
            -theta * df.inner(df.grad(uD), df.grad(phi * v)) * dx
            + theta * df.inner(df.dot(df.grad(uD), n), phi * v) * ds
        )
        L += (
            -(1.0 - theta) * df.inner(df.grad(u_n), df.grad(phi * v)) * dx
            + (1.0 - theta) * df.inner(df.dot(df.grad(u_n), n), phi * v) * ds
        )

        if ghost == True:
            L += -theta * G_h(uD, phi * v) - (1.0 - theta) * G_h(u_n, phi * v)
            L += (
                -sigma
                * h**2
                * df.inner(
                    theta * f + (1.0 - theta) * f,
                    df.div(df.grad(phi * v)),
                )
                * dx(1)
            )
            L += (
                sigma
                * h**2
                * df.inner(
                    dt ** (-1) * (uD - u_n)
                    - theta * df.div(df.grad(uD))
                    - (1 - theta) * df.div(df.grad(u_n)),
                    df.div(df.grad(phi * v)),
                )
                * dx(1)
            )

        w_n1 = df.Function(V)
        df.solve(a == L, w_n1, solver_parameters={"linear_solver": "mumps"})
        u_n = w_n1 * phi + uD
        sol += [u_n]
        print("(", i + 1, ",", ind, "/", len(time) - 1, ")")

    sol_phi_fem = [df.project(sol[j], V_std) for j in range(len(time))]

    if Plot:
        plt.close()

        fig = plt.figure()
        for j in range(len(time)):
            df.plot(sol_phi_fem[j], title=f"Time = {time[j]}")
            if save_figs:
                plt.savefig(f"outputs/images/plot_phi_fem_{j}.png")
            plt.pause(0.1)
            fig.clear()

    if compute_error:

        # Computation of the error
        norm_L2_exact = 0.0
        err_L2 = 0.0
        norm_H1_exact = 0.0
        err_H1 = 0.0
        sol_phi_fem = [df.project(sol[j], V_std) for j in range(len(time))]
        for j in range(len(time)):
            norm_L2_exact_j = df.assemble(sol_std[j] ** 2 * df.dx)
            if norm_L2_exact < norm_L2_exact_j:
                norm_L2_exact = norm_L2_exact_j
            err_L2_j = df.assemble((sol_phi_fem[j] - sol_std[j]) ** 2 * df.dx)
            if err_L2 < err_L2_j:
                err_L2 = err_L2_j
            norm_H1_exact += df.assemble(
                dt * df.grad(sol_std[j]) ** 2 * df.dx
            )
            err_H1 += df.assemble(
                dt * df.grad(sol_phi_fem[j] - sol_std[j]) ** 2 * df.dx
            )
        err_L2 = err_L2**0.5 / norm_L2_exact**0.5
        err_H1 = err_H1**0.5 / norm_H1_exact**0.5
        size_mesh_phi_fem_vec[i] = mesh.hmax()
        error_L2_phi_fem_vec[i] = err_L2
        error_H1_phi_fem_vec[i] = err_H1
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
if write_output and compute_error:
    file.write("relative L2 norm phi fem : \n")
    output_latex(file, size_mesh_phi_fem_vec, error_L2_phi_fem_vec)
    file.write("relative H1 norm phi fem : \n")
    output_latex(file, size_mesh_phi_fem_vec, error_H1_phi_fem_vec)

# Print the output vectors  phi_fem
print("Vector h phi fem:", size_mesh_phi_fem_vec)
print("Vector relative L2 error phi fem : ", error_L2_phi_fem_vec)
print("Vector relative H1 error phi fem : ", error_H1_phi_fem_vec)
if conditioning and write_output:
    file.write("conditionning number phi fem x h^2 : \n")
    output_latex(file, size_mesh_phi_fem_vec, cond_phi_fem_vec)
    print("conditioning number phi_fem", cond_phi_fem_vec)

if write_output:
    file.close()
