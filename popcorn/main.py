import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import mshr
import time
import sympy
import vedo
import vedo.dolfin as vdf
import pygalmesh
import os

plt.style.use("ggplot")

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True

# test case :
# 1) L2H1 error , P^1
# 2) LinfL2 error, P^1
test_case = 1

if test_case == 1:
    # expected slope 1
    # puissance on dt
    exp_dt = 1.0
    # Number of iterations
    Iter = 5
    degV = 1
    exact_mesh_size = 0.013
elif test_case == 2:
    # expected slope 2
    # puissance on dt
    exp_dt = 2.0
    # Number of iterations
    Iter = 5
    degV = 1
    exact_mesh_size = 0.016

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
    "./outputs/output_test_case_{test_case}_sigma_{sigma}.txt".format(
        test_case=test_case, sigma=sigma
    ),
    "w",
)

if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./data/meshes"):
    os.makedirs("./data/meshes")
if not os.path.exists("./data/f"):
    os.makedirs("./data/f")
if not os.path.exists("./data/f_exact"):
    os.makedirs("./data/f_exact")
if not os.path.exists("./data/sol"):
    os.makedirs("./data/sol")
if not os.path.exists("./data/sol_exact"):
    os.makedirs("./data/sol_exact")
if not os.path.exists("./data/f_std"):
    os.makedirs("./data/f_std")
if not os.path.exists("./data/sol_std"):
    os.makedirs("./data/sol_std")


class phi_expr(df.UserExpression):
    def eval(self, value, x):
        r0, sigma, A = 0.6, 0.3, 1.5
        xx, yy, zz = x[0], x[1], x[2]
        phi = (xx**2 + yy**2 + zz**2) - r0**2
        for k in range(0, 12):
            if k >= 0 and k <= 4:
                xk = (r0 / np.sqrt(5)) * 2.0 * np.cos(2.0 * k * np.pi / 5.0)
                yk = (r0 / np.sqrt(5)) * 2.0 * np.sin(2.0 * k * np.pi / 5.0)
                zk = r0 / np.sqrt(5)
            elif k >= 5 and k <= 9:
                xk = (
                    (r0 / np.sqrt(5))
                    * 2.0
                    * np.cos((2.0 * (k - 5.0) - 1.0) * np.pi / 5.0)
                )
                yk = (
                    (r0 / np.sqrt(5))
                    * 2.0
                    * np.sin((2.0 * (k - 5.0) - 1.0) * np.pi / 5.0)
                )
                zk = -(r0 / np.sqrt(5))
            elif k == 10:
                xk = 0.0
                yk = 0.0
                zk = r0
            elif k == 11:
                xk = 0.0
                yk = 0.0
                zk = -r0
            phi -= A * np.exp(
                -((xx - xk) ** 2 + (yy - yk) ** 2 + (zz - zk) ** 2)
                / sigma**2
            )
        value[0] = phi

    def value_shape(self):
        return (2,)


class Popcorn(pygalmesh.DomainBase):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        r0, sigma, A = 0.6, 0.3, 1.5
        xx, yy, zz = x[0], x[1], x[2]
        phi = (xx**2 + yy**2 + zz**2) - r0**2
        for k in range(0, 12):
            if k >= 0 and k <= 4:
                xk = (r0 / np.sqrt(5)) * 2.0 * np.cos(2.0 * k * np.pi / 5.0)
                yk = (r0 / np.sqrt(5)) * 2.0 * np.sin(2.0 * k * np.pi / 5.0)
                zk = r0 / np.sqrt(5)
            elif k >= 5 and k <= 9:
                xk = (
                    (r0 / np.sqrt(5))
                    * 2.0
                    * np.cos((2.0 * (k - 5.0) - 1.0) * np.pi / 5.0)
                )
                yk = (
                    (r0 / np.sqrt(5))
                    * 2.0
                    * np.sin((2.0 * (k - 5.0) - 1.0) * np.pi / 5.0)
                )
                zk = -(r0 / np.sqrt(5))
            elif k == 10:
                xk = 0.0
                yk = 0.0
                zk = r0
            elif k == 11:
                xk = 0.0
                yk = 0.0
                zk = -r0
            phi -= A * np.exp(
                -((xx - xk) ** 2 + (yy - yk) ** 2 + (zz - zk) ** 2)
                / sigma**2
            )

        return phi

    def get_bounding_sphere_squared_radius(self):
        return 10.0


print("###########################")
text = " Exact FEM "
print(f"{text:#^27}")
print("###########################")


if not os.path.exists(f"./data/meshes/popcorn_exact_{exact_mesh_size}.xml"):
    d = Popcorn()
    mesh = pygalmesh.generate_mesh(d, max_cell_circumradius=exact_mesh_size)
    mesh.write(f"./data/meshes/popcorn_exact_{exact_mesh_size}.xml")
mesh_exact = df.Mesh(f"./data/meshes/popcorn_exact_{exact_mesh_size}.xml")

print(f"{mesh_exact.hmax()=}")

V_exact = df.FunctionSpace(mesh_exact, "CG", degV)
# Computation of the source term
t, x, y, z = sympy.symbols("tt xx yy zz")
f1 = sympy.exp(
    -((x - 0.2) ** 2 + (y - 0.3) ** 2 + (z + 0.1) ** 2) / (2.0 * 0.3**2)
)

# Define boundary condition
bc_exact = df.DirichletBC(V_exact, df.Constant(0.0), "on_boundary")
dx_exact = df.Measure("dx")(domain=mesh_exact)
dt_exact = mesh_exact.hmax() ** exp_dt
Time_exact = np.arange(0.0, T, dt_exact)

# f_exact = []
index_temps = 0
for temps in Time_exact:
    f_exact = df.project(
        df.Expression(
            sympy.ccode(f1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]")
            .replace("tt", "temps"),
            temps=temps,
            degree=degV + 1,
            domain=mesh_exact,
        ),
        V_exact,
        solver_type="gmres",
        preconditioner_type="hypre_amg",
    )

    np.save(
        f"./data/f_exact/f_exact_{index_temps}.npy",
        f_exact.vector().get_local(),
    )
    index_temps += 1

# Resolution
u_exact = df.TrialFunction(V_exact)
v_exact = df.TestFunction(V_exact)
u_n_exact = df.Expression("0.0", degree=degV, domain=mesh_exact)
sol_exact = df.project(
    u_n_exact, V_exact, solver_type="gmres", preconditioner_type="hypre_amg"
)
np.save(f"./data/sol_exact/sol_exact_{0}.npy", sol_exact.vector().get_local())
a_exact = (
    dt_exact ** (-1) * u_exact * v_exact * dx_exact
    + df.inner(df.grad(u_exact), df.grad(v_exact)) * dx_exact
)
for ind in range(1, len(Time_exact)):
    f_exact_array = np.load(f"./data/f_exact/f_exact_{ind}.npy")
    loaded_f_exact = df.Function(V_exact)
    loaded_f_exact.vector()[:] = f_exact_array
    L_exact = (
        dt_exact ** (-1) * u_n_exact * v_exact * dx_exact
        + loaded_f_exact * v_exact * dx_exact
    )
    u_n1_exact = df.Function(V_exact)
    df.solve(
        a_exact == L_exact,
        u_n1_exact,
        bc_exact,
        solver_parameters={
            "linear_solver": "gmres",
            "preconditioner": "hypre_amg",
        },
    )
    sol_exact = df.project(
        u_n1_exact,
        V_exact,
        solver_type="gmres",
        preconditioner_type="hypre_amg",
    )
    np.save(
        f"./data/sol_exact/sol_exact_{ind}.npy", sol_exact.vector().get_local()
    )
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

size = 1
for i in range(0, Iter):
    print("###########################")
    text = " Iteration phi FEM " + str(i + 1) + " "
    print(f"{text:#^27}")
    print("###########################")

    # Construction of the mesh
    dt = dt_exact * (2 ** (Iter - i))
    print(f"{dt_exact=}")
    print(f"{dt=}")
    Time = np.arange(0, T, dt)

    hmax = 1.0e4
    while hmax > (dt + df.DOLFIN_EPS) ** (1.0 / exp_dt):
        mesh_macro = df.BoxMesh(
            df.Point(-2.0, -2.0, -2.0),
            df.Point(2.0, 2.0, 2.0),
            size,
            size,
            size,
        )
        size += 1
        hmax = mesh_macro.hmax()
    print(f"{hmax**exp_dt=}    {size=}     {dt=}")

    V_phi = df.FunctionSpace(mesh_macro, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    domains = df.MeshFunction(
        "size_t", mesh_macro, mesh_macro.topology().dim()
    )
    domains.set_all(0)
    for ind in range(mesh_macro.num_cells()):
        mycell = df.Cell(mesh_macro, ind)
        (
            v1x,
            v1y,
            v1z,
            v2x,
            v2y,
            v2z,
            v3x,
            v3y,
            v3z,
            v4x,
            v4y,
            v4z,
        ) = mycell.get_vertex_coordinates()
        if (
            phi(v1x, v1y, v1z) <= 0.0
            or df.near(phi(v1x, v1y, v1z), 0.0)
            or phi(v2x, v2y, v2z) <= 0.0
            or df.near(phi(v2x, v2y, v2z), 0.0)
            or phi(v3x, v3y, v3z) <= 0.0
            or df.near(phi(v3x, v3y, v3z), 0.0)
            or phi(v4x, v4y, v4z) <= 0.0
            or df.near(phi(v4x, v4y, v4z), 0.0)
        ):
            domains[ind] = 1
    mesh = df.SubMesh(mesh_macro, domains, 1)

    print("num cell mesh:", mesh.num_cells())
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    V = df.FunctionSpace(mesh, "CG", degV)

    # Facets and cells where we apply the ghost penalty
    mesh.init(1, 2)
    facet_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    cell_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    facet_ghost.set_all(0)
    cell_ghost.set_all(0)
    for mycell in df.cells(mesh):
        for myfacet in df.facets(mycell):
            v1, v2, v3 = df.vertices(myfacet)
            if (
                phi(v1.point().x(), v1.point().y(), v1.point().z())
                * phi(v2.point().x(), v2.point().y(), v2.point().z())
                < 0.0
                or phi(v1.point().x(), v1.point().y(), v1.point().z())
                * phi(v3.point().x(), v3.point().y(), v3.point().z())
                < 0.0
                or phi(v2.point().x(), v2.point().y(), v2.point().z())
                * phi(v3.point().x(), v3.point().y(), v3.point().z())
                < 0.0
            ):
                cell_ghost[mycell] = 1
                for myfacet2 in df.facets(mycell):
                    facet_ghost[myfacet2] = 1

    # Computation of the source term and exact solution

    f_expr = []
    index_temps = 0
    for temps in Time:
        f_expr = df.project(
            df.Expression(
                sympy.ccode(f1)
                .replace("xx", "x[0]")
                .replace("yy", "x[1]")
                .replace("zz", "x[2]")
                .replace("tt", "temps"),
                temps=temps,
                degree=degV + 1,
                domain=mesh,
            ),
            V,
            solver_type="gmres",
            preconditioner_type="hypre_amg",
        )

        np.save(
            f"./data/f/f_expr_{index_temps}.npy", f_expr.vector().get_local()
        )
        index_temps += 1

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
    sol = phi * wn
    sol = df.project(
        sol, V, solver_type="gmres", preconditioner_type="hypre_amg"
    )
    np.save(f"./data/sol/sol_{0}.npy", sol.vector().get_local())
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
        f_expr_array = np.load(f"./data/f/f_expr_{ind}.npy")
        loaded_f_expr = df.Function(V)
        loaded_f_expr.vector()[:] = f_expr_array

        L = loaded_f_expr * phi * v * dx + dt ** (-1) * phi * wn * phi * v * dx
        if ghost == True:
            L += (
                -sigma
                * h**2
                * df.inner(loaded_f_expr, df.div(df.grad(phi * v)))
                * dx(1)
            )
            L += (
                -sigma
                * h**2
                * df.inner(dt ** (-1) * phi * wn, df.div(df.grad(phi * v)))
                * dx(1)
            )

        w_n1 = df.Function(V)

        df.solve(
            a == L,
            w_n1,
            solver_parameters={
                "linear_solver": "gmres",
                "preconditioner": "hypre_amg",
            },
        )

        wn = w_n1
        sol = df.project(
            phi * wn, V, solver_type="gmres", preconditioner_type="hypre_amg"
        )
        np.save(f"./data/sol/sol_{ind}.npy", sol.vector().get_local())
        print("(", i + 1, ",", ind, "/", len(Time) - 1, ")")
    # Computation of the error
    norm_L2_exact = 0.0
    err_L2 = 0.0
    norm_H1_exact = 0.0
    err_H1 = 0.0
    j_exact = 0
    for j in range(len(Time)):
        sol_exact_array = np.load(
            f"./data/sol_exact/sol_exact_{int(2 ** (Iter - i) * j)}.npy"
        )
        sol_exactj = df.Function(V_exact)
        sol_exactj.vector()[:] = sol_exact_array

        sol_array = np.load(f"./data/sol/sol_{j}.npy")
        solj = df.Function(V)
        solj.vector()[:] = sol_array

        Iu_hj = df.project(
            solj, V_exact, solver_type="gmres", preconditioner_type="hypre_amg"
        )
        norm_L2_exact_j = df.assemble(sol_exactj**2 * dx_exact)
        if norm_L2_exact < norm_L2_exact_j:
            norm_L2_exact = norm_L2_exact_j
        err_L2_j = df.assemble((Iu_hj - sol_exactj) ** 2 * dx_exact)
        if err_L2 < err_L2_j:
            err_L2 = err_L2_j
        norm_H1_exact += df.assemble(dt * df.grad(sol_exactj) ** 2 * dx_exact)
        err_H1 += df.assemble(dt * df.grad(Iu_hj - sol_exactj) ** 2 * dx_exact)
        print(f"Time : {j=}   {Time_exact[2**(Iter-i)*j]=}   {Time[j]=}")

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


size_mesh_standard_vec = np.zeros(Iter)
error_L2_standard_vec = np.zeros(Iter)
error_H1_standard_vec = np.zeros(Iter)

size = 0.6
for i in range(0, Iter):
    print("###########################")
    text = " Iteration standard " + str(i + 1) + " "
    print(f"{text:#^27}")
    print("###########################")

    dt = dt_exact * (2 ** (Iter - i))
    print(f"{dt_exact=}")
    print(f"{dt=}")
    Time = np.arange(0, T, dt)
    # Construction of the mesh
    hmax = 1.0e4

    if not os.path.exists(
        f"./data/meshes/popcorn_{exact_mesh_size}_iter_{i}.xml"
    ):
        d = Popcorn()
        while hmax > (dt + df.DOLFIN_EPS) ** (1.0 / exp_dt):
            mesh = pygalmesh.generate_mesh(d, max_cell_circumradius=size)
            mesh.write(f"./data/meshes/popcorn_{exact_mesh_size}_iter_{i}.xml")
            mesh = df.Mesh(
                f"./data/meshes/popcorn_{exact_mesh_size}_iter_{i}.xml"
            )
            size -= 0.001
            hmax = mesh.hmax()
    else:
        mesh = df.Mesh(f"./data/meshes/popcorn_{exact_mesh_size}_iter_{i}.xml")
        hmax = mesh.hmax()

    print(f"{hmax**exp_dt=}    {size=}     {dt=}")

    V = df.FunctionSpace(mesh, "CG", degV)

    # Computation of the source term
    uD = df.Expression("0.0", degree=degV, domain=mesh)
    bc = df.DirichletBC(V, df.Constant(0.0), "on_boundary")

    index_temps = 0
    for temps in Time:
        f_std = df.project(
            df.Expression(
                sympy.ccode(f1)
                .replace("xx", "x[0]")
                .replace("yy", "x[1]")
                .replace("zz", "x[2]")
                .replace("tt", "temps"),
                temps=temps,
                degree=degV + 1,
                domain=mesh,
            ),
            V,
            solver_type="gmres",
            preconditioner_type="hypre_amg",
        )

        np.save(
            f"./data/f_std/f_std_{index_temps}.npy", f_std.vector().get_local()
        )
        index_temps += 1

    # Resolution
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    dx = df.Measure("dx", domain=mesh)
    u_n = df.Expression("0.0", degree=degV, domain=mesh)
    sol_std = df.project(
        u_n, V, solver_type="gmres", preconditioner_type="hypre_amg"
    )
    np.save(f"./data/sol_std/sol_std_{0}.npy", sol_std.vector().get_local())
    a = dt ** (-1) * u * v * dx + df.inner(df.grad(u), df.grad(v)) * dx
    for ind in range(1, len(Time)):
        f_std_array = np.load(f"./data/f_std/f_std_{ind}.npy")
        loaded_f_std = df.Function(V)
        loaded_f_std.vector()[:] = f_std_array
        L = dt ** (-1) * u_n * v * dx + loaded_f_std * v * dx
        u_n1 = df.Function(V)
        df.solve(
            a == L,
            u_n1,
            bc,
            solver_parameters={
                "linear_solver": "gmres",
                "preconditioner": "hypre_amg",
            },
        )
        sol_std = df.project(
            u_n1, V, solver_type="gmres", preconditioner_type="hypre_amg"
        )
        np.save(
            f"./data/sol_std/sol_std_{ind}.npy", sol_std.vector().get_local()
        )
        u_n = u_n1
        print("(", ind, "/", len(Time) - 1, ")")

    # Computation of the error
    norm_L2_exact = 0.0
    err_L2 = 0.0
    norm_H1_exact = 0.0
    err_H1 = 0.0
    j_exact = 0
    for j in range(len(Time)):
        sol_exact_array = np.load(
            f"./data/sol_exact/sol_exact_{int(2 ** (Iter - i) * j)}.npy"
        )
        sol_exactj = df.Function(V_exact)
        sol_exactj.vector()[:] = sol_exact_array

        sol_array = np.load(f"./data/sol_std/sol_std_{j}.npy")
        solj = df.Function(V)
        solj.vector()[:] = sol_array

        Iu_hj = df.project(
            solj, V_exact, solver_type="gmres", preconditioner_type="hypre_amg"
        )
        norm_L2_exact_j = df.assemble(sol_exactj**2 * dx_exact)
        if norm_L2_exact < norm_L2_exact_j:
            norm_L2_exact = norm_L2_exact_j
        err_L2_j = df.assemble((Iu_hj - sol_exactj) ** 2 * dx_exact)
        if err_L2 < err_L2_j:
            err_L2 = err_L2_j
        norm_H1_exact += df.assemble(dt * df.grad(sol_exactj) ** 2 * dx_exact)
        err_H1 += df.assemble(dt * df.grad(Iu_hj - sol_exactj) ** 2 * dx_exact)
        print(f"Time : {j=}   {Time_exact[2**(Iter-i)*j]=}   {Time[j]=}")
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

print("Size cell mesh exact:", mesh_exact.hmax())
# Print the output vectors  phi_fem
print("Vector h phi fem:", size_mesh_phi_fem_vec)
print("Vector relative L2 error phi fem : ", error_L2_phi_fem_vec)
print("Vector relative H1 error phi fem : ", error_H1_phi_fem_vec)


# Print the output vectors  standard fem
print("Vector h standard fem:", size_mesh_standard_vec)
print("Vector relative L2 error standard fem : ", error_L2_standard_vec)
print("Vector relative H1 error standard fem : ", error_H1_standard_vec)

if write_output:
    f.close()


plt.figure()
if test_case == 1 or test_case == 3:
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

elif test_case == 2 or test_case == 4:
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

plt.tight_layout()
plt.savefig(f"outputs/plot_problem_test_case_{test_case}_sigma_{sigma}.png")
plt.show()
