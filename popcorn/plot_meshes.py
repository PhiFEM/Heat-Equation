import dolfin as df
import mshr
import pygalmesh
import vedo
import vedo.dolfin as vdf
import numpy as np
import os

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True

filenames = ["exact", "phi_fem", "standard"]

if not os.path.exists("./meshes"):
    os.makedirs("./meshes")

if not os.path.exists("./images"):
    os.makedirs("./images")

for name in filenames:
    if name == "exact":

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
                        -((xx - xk) ** 2 + (yy - yk) ** 2 + (zz - zk) ** 2) / sigma**2
                    )

                return phi

            def get_bounding_sphere_squared_radius(self):
                return 10.0

        if not os.path.exists(f"./meshes/popcorn_exact_domain.xml"):
            d = Popcorn()
            mesh = pygalmesh.generate_mesh(d, max_cell_circumradius=0.02)
            mesh.write("./meshes/popcorn_exact_domain.xml")
        mesh = df.Mesh("./meshes/popcorn_exact_domain.xml")
        vdf.plot(mesh, lw=0, axes=0, c="gray")
        vedo.screenshot("./images/domain.png")
        vedo.close()

    elif name == "phi_fem":

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
                        -((xx - xk) ** 2 + (yy - yk) ** 2 + (zz - zk) ** 2) / sigma**2
                    )
                value[0] = phi

            def value_shape(self):
                return (2,)

        mesh_macro = df.BoxMesh(
            df.Point(-1.0, -1.0, -1.0), df.Point(1.0, 1.0, 1.0), 20, 20, 20
        )

        V_phi = df.FunctionSpace(mesh_macro, "CG", 2)
        phi = phi_expr(element=V_phi.ufl_element())
        phi = df.interpolate(phi, V_phi)
        domains = df.MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
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
        vdf.plot(mesh, axes=0, c="yellow")
        vedo.screenshot("./images/phi_fem_mesh.png")
        vedo.close()

    elif name == "standard":

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
                        -((xx - xk) ** 2 + (yy - yk) ** 2 + (zz - zk) ** 2) / sigma**2
                    )

                return phi

            def get_bounding_sphere_squared_radius(self):
                return 10.0

        if not os.path.exists(f"./meshes/popcorn_plot_standard.xml"):
            d = Popcorn()
            mesh = pygalmesh.generate_mesh(d, max_cell_circumradius=0.1)
            mesh.write("./meshes/popcorn_plot_standard.xml")
        mesh = df.Mesh("./meshes/popcorn_plot_standard.xml")
        vdf.plot(mesh, axes=0, c="yellow")
        vedo.screenshot("./images/standard_mesh.png")
        vedo.close()
