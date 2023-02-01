import dolfin as df
import numpy as np
import sympy
import vedo
import vedo.dolfin as vdf
from vedo.shapes import Circle, Disc
import mshr
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


domain_mesh = mshr.Circle(df.Point(0.0, 0.0), 1.0)  # creation of the domain
print("\n")
print("###########################")
text = " Mesh Standard FEM "
print(f"{text:#^27}")
print("###########################")
# Construction of the mesh
N = int(8 * 2 ** (i - 1))
mesh = mshr.generate_mesh(domain_mesh, N)
vdf.plot(mesh, c="yellow", interactive=True, axes=0)
vedo.screenshot("outputs/standard_mesh.png")
vedo.close()
