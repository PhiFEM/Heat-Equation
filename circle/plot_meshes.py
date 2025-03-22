import dolfin as df
import vedo
import vedo.dolfin as vdf
from vedo.shapes import Disc
import numpy as np
import pygalmesh
import os

df.parameters["allow_extrapolation"] = True

# Polynome Pk
polV = 1
degPhi = 1 + polV

if not os.path.exists("./outputs"):
    os.makedirs("./outputs")

######################
#    Exact domain    #
######################
print("###########################")
text = " Exact domain "
print(f"{text:#^27}")
print("###########################")

circle = Disc(pos=(0.0, 0.0), r1=1.0, r2=1.0 + 0.004, c="black")
circle_inside = Disc(pos=(0.0, 0.0), r1=0.0, r2=1.0, c="gray")
Omega = r"\Omega"
formula = vedo.Latex(Omega, c="k", s=0.3, usetex=False, res=60).pos(
    -0.25, -0.25, 0
)
vedo.show(circle, circle_inside, formula)
vedo.screenshot("outputs/domain.png")
vedo.close()

i = 1
print("\n")
print("###########################")
text = " Mesh Phi-FEM "
print(f"{text:#^27}")
print("###########################")

# Construction of the mesh
N = int(40 * 2 ** ((i)))
mesh_macro = df.RectangleMesh(df.Point(-2.0, -2.0), df.Point(2.0, 2.0), N, N)
V_phi = df.FunctionSpace(mesh_macro, "CG", degPhi)
phi = df.Expression(
    "-1. +pow(x[0],2)+pow(x[1],2)", degree=degPhi, domain=mesh_macro
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
vdf.plot(mesh, c="yellow", interactive=False, axes=0)
vdf.plot(circle, c="black", interactive=True, add=True, axes=0)
vedo.screenshot("outputs/phi_fem_mesh.png")
vedo.close()


print("\n")
print("###########################")
text = " Mesh Standard FEM "
print(f"{text:#^27}")
print("###########################")
# Construction of the mesh
N = int(8 * 2 ** (i - 1))
points = np.array(
    [
        [np.cos(alpha), np.sin(alpha)]
        for alpha in np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    ]
)
constraints = [[k, k + 1] for k in range(N - 1)] + [[N - 1, 0]]

max_edge_size = 2. * np.pi / (3. * N)
pyg_mesh = pygalmesh.generate_2d(points, constraints, max_edge_size=max_edge_size, num_lloyd_steps=10)
pyg_mesh.write(f"./data/meshes/circle_{str(i).zfill(2)}.xml")
mesh = df.Mesh(f"./data/meshes/circle_{str(i).zfill(2)}.xml")
vdf.plot(mesh, c="yellow", interactive=True, axes=0)
vedo.screenshot("outputs/standard_mesh.png")
vedo.close()
