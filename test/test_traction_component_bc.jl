using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

vbasis = TensorProductBasis(2,1)
sbasis = TensorProductBasis(1,1)
vquad = tensor_product_quadrature(2,2)
squad = tensor_product_quadrature(1,2)
facequads = repeat([squad],4)
facemaps = HDGElasticity.reference_cell_facemaps(2)
normals = HDGElasticity.reference_normals()
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.,2.))
cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
facescale = HDGElasticity.face_determinant_jacobian(cellmap)
stabilization = 1.0


locop = HDGElasticity.local_operator(vbasis,vquad,facequads,facemaps,
    Dhalf,stabilization,cellmap)

lochyb = HDGElasticity.local_hybrid_operator(vbasis,sbasis,facequads,facemaps,
    normals,Dhalf,stabilization,cellmap)
HH = HDGElasticity.HHop(sbasis,facequads,stabilization,cellmap)

components = [1.0,0.0]
faceid = 1
hybloc1 = HDGElasticity.hybrid_local_operator_traction_components(sbasis,
    vbasis,facequads[faceid],facemaps[faceid],normals[faceid],
    components,Dhalf,stabilization,facescale[faceid])
HHc1 = HDGElasticity.HHop(sbasis,squad,components,facescale[faceid])


# v2 = Alh[2]'*(Klop.lulop\Alh[2]) - Ahh[2]
# R = HDGElasticity.linear_form([1.0,0.0],sbasis,squad)
#
# v3 = Alh[3]'*(Klop.lulop\Alh[3]) - Ahh[3]
#
# comp = [0.0,1.0]
# normal = [-1.0,0.0]
# Ahl = HDGElasticity.hybrid_local_operator(sbasis,vbasis,squad,comp,normal,
#     Dhalf,4,cellmap,1.0)
# Ahh4 = HDGElasticity.HHop(sbasis,squad,comp,4,cellmap,alpha)
# Ahh4bc = HDGElasticity.HHop(sbasis,squad,normal,4,cellmap,1.0)
# v4 = Ahl*(Klop.lulop\Alh[4]) - Ahh4 + Ahh4bc
