import dolfin as df 
import mshr
from vedo.dolfin import plot, screenshot
import numpy as np 
import sympy 

df.parameters["allow_extrapolation"] = True

# Choose the type of simulation to do 
compute_phi_fem = True 
compute_standard = True

write_outputs = True

# Time of simulation
T = 5.0

# Number of iterations
init_Iter = 0
Iter = 4

# parameter of the ghost penalty
sigma = 20.0

# Plot meshes for each iterations 
Plot = True

# Polynome Pk
polV = 1
degPhi = 2 + polV

# Ghost penalty
ghost = True

# Compute the conditioning number
conditioning = False

# Parameter for theta-scheme
theta = 1.0

# Function used to write in the outputs files
def output_latex(f, A, B):
    for i in range(len(A)):
        f.write("(")
        f.write(str(A[i]))
        f.write(",")
        f.write(str(B[i]))
        f.write(")\n")
    f.write("\n")


# Computation of the exact solution and exact source term
t, x, y, z = sympy.symbols("tt xx yy zz")
u1 = sympy.exp(x) * sympy.sin(2 * sympy.pi * y)* sympy.sin(2 * sympy.pi * z) * sympy.sin(t)
f1 = (
    sympy.diff(u1, t)
    - sympy.diff(sympy.diff(u1, x), x)
    - sympy.diff(sympy.diff(u1, y), y)
    - sympy.diff(sympy.diff(u1, z), z)
)

if write_outputs : 
    if compute_standard and not(compute_phi_fem):
        f = open("outputs/3D_P{deg}_theta_{theta}_std.txt".format(deg=polV, theta=theta), "w")
    elif compute_phi_fem and not(compute_standard):
        f = open("outputs/3D_P{deg}_theta_{theta}_phi_fem.txt".format(deg=polV, theta=theta), "w")
    elif compute_standard and compute_phi_fem:
        f = open("outputs/3D_P{deg}_theta_{theta}.txt".format(deg=polV, theta=theta), "w")

    f.write("theta = " + str(theta) + "\n")
#Initialistion of the output

if compute_phi_fem:
    size_mesh_phi_fem_vec = np.zeros(Iter)
    error_L2_phi_fem_vec = np.zeros(Iter)
    error_H1_phi_fem_vec = np.zeros(Iter)
    cond_phi_fem_vec = np.zeros(Iter)

    for i in range(init_Iter, Iter):
        print("###########################")
        print("## Iteration phi FEM ", i + 1, "##")
        print("###########################")

        # Construction of the mesh
        N = int(10 * 2 ** ((i-1)))
        mesh_macro = df.UnitCubeMesh(N, N, N)
        dt = mesh_macro.hmax() #10.0*mesh_macro.hmax()**2
        time_ = np.arange(0, T + dt, dt)
        V_phi = df.FunctionSpace(mesh_macro, "CG", degPhi)
        phi = df.Expression(
            "-0.125+pow(x[0]-0.5,2)+pow(x[1]-0.5,2)+pow(x[2]-0.5,2)",
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
            v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z = mycell.get_vertex_coordinates()
            if phi(v1x, v1y, v1z) <= 0 or phi(v2x, v2y, v2z) <= 0 or phi(v3x, v3y, v3z) <= 0 or phi(v4x, v4y, v4z) <= 0:
                domains[ind] = 1
        mesh = df.SubMesh(mesh_macro, domains, 1)
        V = df.FunctionSpace(mesh, "CG", polV)

        # Construction of phi
        V_phi = df.FunctionSpace(mesh, "CG", degPhi)
        phi = df.Expression(
            "-0.125+pow(x[0]-0.5,2)+pow(x[1]-0.5,2)+pow(x[2]-0.5,2)", degree=degPhi, domain=mesh
        )
        phi = df.interpolate(phi, V_phi)

        if Plot :
            plot(mesh, axes=0)
            screenshot(f"outputs/figs/mesh_phi_fem_h_{mesh.hmax()}.png")
        # Computation of the source term and exact solution
        f_expr = []
        u_expr = []
        for temps in time_:
            f_expr += [
                df.Expression(
                    sympy.ccode(f1)
                    .replace("xx", "x[0]")
                    .replace("yy", "x[1]")
                    .replace("zz", "x[2]")
                    .replace("tt", "temps"),
                    temps=temps,
                    degree=polV + 1,
                    domain=mesh,
                )
            ]
            u_expr += [
                df.Expression(
                    sympy.ccode(u1)
                    .replace("xx", "x[0]")
                    .replace("yy", "x[1]")
                    .replace("zz", "x[2]")
                    .replace("tt", "temps"),
                    temps=temps,
                    degree=4,
                    domain=mesh,
                )
            ]

        # Facets and cells where we apply the ghost penalty
        mesh.init(1, 2)
        facet_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        cell_ghost = df.MeshFunction("size_t", mesh, mesh.topology().dim())
        facet_ghost.set_all(0)
        cell_ghost.set_all(0)
        for cell in df.cells(mesh) : 
            for facet in df.facets(cell): 
                v1,v2,v3 = df.vertices(facet) 
                if(phi(v1.point())*phi(v2.point()) < 0.0 or phi(v1.point())*phi(v3.point()) < 0.0 or phi(v2.point())*phi(v3.point()) < 0.0): 
                    cell_ghost[cell] = 1  
                    for facett in df.facets(cell):
                        facet_ghost[facett] = 1 

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
            sympy.ccode(u1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]")
            .replace("tt", "0.0"),
            degree=polV + 2,
            domain=mesh,
        )
        sol = [u_n]
        a = (
            dt ** (-1) * phi * w * phi * v * dx
            + theta * df.inner(df.grad(phi * w), df.grad(phi * v)) * dx
            - theta * df.dot(df.inner(df.grad(phi * w), n), phi * v) * ds
        )

        if ghost == True:
            a += theta * G_h(phi * w, phi * v) - sigma * h ** 2 * df.inner(
                phi * w * dt ** (-1) - theta * df.div(df.grad(phi * w)),
                df.div(df.grad(phi * v)),
            ) * dx(1)

        for ind in range(1, len(time_)):
            uD_n = (u_expr[ind]) * (1.0 + phi)

            L = (
                (theta * f_expr[ind] + (1.0 - theta) * f_expr[ind - 1])
                * phi
                * v
                * dx
                + dt ** (-1) * (u_n) * phi * v * dx
                - dt ** (-1) * uD_n * phi * v * dx
            )
            L += (
                -theta * df.inner(df.grad(uD_n), df.grad(phi * v)) * dx
                + theta * df.inner(df.dot(df.grad(uD_n), n), phi * v) * ds
            )
            L += (
                -(1.0 - theta) * df.inner(df.grad(u_n), df.grad(phi * v)) * dx
                + (1.0 - theta) * df.inner(df.dot(df.grad(u_n), n), phi * v) * ds
            )

            if ghost == True:
                L += -theta * G_h(uD_n, phi * v) - (1.0 - theta) * G_h(
                    u_n, phi * v
                )
                L += (
                    -sigma
                    * h ** 2
                    * df.inner(
                        theta * f_expr[ind] + (1 - theta) * f_expr[ind - 1],
                        df.div(df.grad(phi * v)),
                    )
                    * dx(1)
                )
                L += (
                    sigma
                    * h ** 2
                    * df.inner(
                        dt ** (-1) * (uD_n - u_n)
                        - theta * df.div(df.grad(uD_n))
                        - (1 - theta) * df.div(df.grad(u_n)),
                        df.div(df.grad(phi * v)),
                    )
                    * dx(1)
                )

            w_n1 = df.Function(V)
            
            df.solve(a == L, w_n1, solver_parameters={"linear_solver": "mumps"})
            u_n = w_n1 * phi + uD_n
            sol += [u_n]
            print("(", i + 1, ",", ind, "/", len(time_) - 1, ")")

        # Computation of the error
        norm_L2_exact = 0.0
        err_L2 = 0.0
        norm_H1_exact = 0.0
        err_H1 = 0.0
        for j in range(len(time_)):
            norm_L2_exact_j = df.assemble(u_expr[j] ** 2 * dx)
            if norm_L2_exact < norm_L2_exact_j:
                norm_L2_exact = norm_L2_exact_j
            err_L2_j = df.assemble((sol[j] - u_expr[j]) ** 2 * dx)
            if err_L2 < err_L2_j:
                err_L2 = err_L2_j
            norm_H1_exact += df.assemble(dt * df.grad(u_expr[j]) ** 2 * dx)
            err_H1 += df.assemble(dt * df.grad(sol[j] - u_expr[j]) ** 2 * dx)
        err_L2 = err_L2 ** 0.5 / norm_L2_exact ** 0.5
        err_H1 = err_H1 ** 0.5 / norm_H1_exact ** 0.5
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
    if write_outputs:
        f.write("relative L2 norm phi fem : \n")
        output_latex(f, size_mesh_phi_fem_vec, error_L2_phi_fem_vec)
        f.write("relative H1 norm phi fem : \n")
        output_latex(f, size_mesh_phi_fem_vec, error_H1_phi_fem_vec)

if compute_standard:
    size_mesh_standard_vec = np.zeros(Iter)
    error_L2_standard_vec = np.zeros(Iter)
    error_H1_standard_vec = np.zeros(Iter)
    cond_standard_vec = np.zeros(Iter)
    domain_mesh = mshr.Sphere(
        df.Point(0.5, 0.5, 0.5),  np.sqrt(2.0) / 4.0
        )  # creation of the domain
    for i in range(init_Iter, Iter):
        print("###########################")
        print("## Iteration standard ", i + 1, "##")
        print("###########################")

        # Construction of the mesh
        N = int(12 * 2 ** (i))
        mesh = mshr.generate_mesh(domain_mesh, N)
        if Plot:
            plot(mesh, axes=0)
            screenshot(f"outputs/figs/mesh_standard_fem_h_{mesh.hmax()}.png")        
        dt = mesh.hmax() #10.0*mesh.hmax()**2
        time_ = np.arange(0, T + dt, dt)
        V = df.FunctionSpace(mesh, "CG", polV)

        # Construction of phi
        V_phi = df.FunctionSpace(mesh, "CG", degPhi)
        phi = df.Expression(
            "-0.125+pow(x[0]-0.5,2)+pow(x[1]-0.5,2)+pow(x[2]-0.5,2)", degree=degPhi, domain=mesh
        )
        phi = df.interpolate(phi, V_phi)

        # Computation of the source term

        f_expr, u_expr = [], []
        for temps in time_:
            f_expr = f_expr + [
                df.Expression(
                    sympy.ccode(f1)
                    .replace("xx", "x[0]")
                    .replace("yy", "x[1]")
                    .replace("zz", "x[2]")
                    .replace("tt", "temps"),
                    temps=temps,
                    degree=polV + 1,
                    domain=mesh,
                )
            ]
            u_expr = u_expr + [
                df.Expression(
                    sympy.ccode(u1)
                    .replace("xx", "x[0]")
                    .replace("yy", "x[1]")
                    .replace("zz", "x[2]")
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
        u_n = df.Expression(
            sympy.ccode(u1)
            .replace("xx", "x[0]")
            .replace("yy", "x[1]")
            .replace("zz", "x[2]")
            .replace("tt", "0.0"),
            degree=polV + 2,
            domain=mesh,
        )
        sol = [u_n]
        a = dt ** (-1) * u * v * dx + theta * df.inner(df.grad(u), df.grad(v)) * dx
        for ind in range(1, len(time_)):
            uD = u_expr[ind] * (1.0 + phi)
            bc = df.DirichletBC(V, uD, "on_boundary")
            L = (
                dt ** (-1) * u_n * v * dx
                + theta * f_expr[ind] * v * dx
                + (1 - theta) * f_expr[ind - 1] * v * dx
                - (1 - theta) * df.inner(df.grad(u_n), df.grad(v)) * dx
            )
            u_n1 = df.Function(V)
            df.solve(
                a == L, u_n1, bc, solver_parameters={"linear_solver": "mumps"}
            )      
            sol = sol + [u_n1]
            u_n = u_n1
            print("(", i + 1, ",", ind, "/", len(time_) - 1, ")")
        # Computation of the error
        norm_L2_exact = 0.0
        err_L2 = 0.0
        norm_H1_exact = 0.0
        err_H1 = 0.0
        for j in range(len(time_)):
            norm_L2_exact_j = df.assemble(u_expr[j] ** 2 * dx)
            if norm_L2_exact < norm_L2_exact_j:
                norm_L2_exact = norm_L2_exact_j
            err_L2_j = df.assemble((sol[j] - u_expr[j]) ** 2 * dx)
            if err_L2 < err_L2_j:
                err_L2 = err_L2_j
            norm_H1_exact += df.assemble(dt * df.grad(u_expr[j]) ** 2 * dx)
            err_H1 += df.assemble(dt * df.grad(sol[j] - u_expr[j]) ** 2 * dx)
        err_L2 = err_L2 ** 0.5 / norm_L2_exact ** 0.5
        err_H1 = err_H1 ** 0.5 / norm_H1_exact ** 0.5
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
    f.write("relative L2 norm standard fem : \n")
    output_latex(f, size_mesh_standard_vec, error_L2_standard_vec)
    f.write("relative H1 norm standard fem : \n")
    output_latex(f, size_mesh_standard_vec, error_H1_standard_vec)


# Print the output vectors  phi_fem
if compute_phi_fem:
    print("Vector h phi fem:", size_mesh_phi_fem_vec)
    print("Vector relative L2 error phi fem : ", error_L2_phi_fem_vec)
    print("Vector relative H1 error phi fem : ", error_H1_phi_fem_vec)
    if conditioning:
        f.write("conditionning number phi fem x h^2 : \n")
        output_latex(f, size_mesh_phi_fem_vec, cond_phi_fem_vec)
        print("conditioning number phi_fem", cond_phi_fem_vec)

# Print the output vectors  standard fem
if compute_standard:
    print("Vector h standard fem:", size_mesh_standard_vec)
    print("Vector relative L2 error standard fem : ", error_L2_standard_vec)
    print("Vector relative H1 error standard fem : ", error_H1_standard_vec)
    if conditioning:
        f.write("conditionning number standard fem x h^2 : \n")
        output_latex(f, size_mesh_standard_vec, cond_standard_vec)
        print(f"conditioning number standard fem : {cond_standard_vec}")

f.close()

