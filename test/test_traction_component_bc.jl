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
Dhalf = HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.,2.)
cellmap = HDGElasticity.AffineMap([0.,0.],[2.,1.])
alpha = 1.0

sysmatrix = HDGElasticity.SystemMatrix()

Klop = HDGElasticity.LocalOperator(vbasis,vquad,squad,Dhalf,cellmap,alpha)

Alh = HDGElasticity.local_hybrid_operator(vbasis,sbasis,squad,Dhalf,cellmap,alpha)
Ahh = HDGElasticity.HHop(sbasis,squad,cellmap,alpha)

comp = [1.0,0.0]
normal = [0.0,-1.0]
Ahl = HDGElasticity.hybrid_local_operator(sbasis,vbasis,squad,comp,normal,
    Dhalf,1,cellmap,1.0)
Ahh1 = HDGElasticity.HHop(sbasis,squad,comp,1,cellmap,alpha)
Ahh1bc = HDGElasticity.HHop(sbasis,squad,normal,1,cellmap,1.0)
v1 = Ahl*(Klop.lulop\Alh[1]) - Ahh1

v2 = Alh[2]'*(Klop.lulop\Alh[2]) - Ahh[2]
R = HDGElasticity.linear_form([1.0,0.0],sbasis,squad)

v3 = Alh[3]'*(Klop.lulop\Alh[3]) - Ahh[3]

comp = [0.0,1.0]
normal = [-1.0,0.0]
Ahl = HDGElasticity.hybrid_local_operator(sbasis,vbasis,squad,comp,normal,
    Dhalf,4,cellmap,1.0)
Ahh4 = HDGElasticity.HHop(sbasis,squad,comp,4,cellmap,alpha)
Ahh4bc = HDGElasticity.HHop(sbasis,squad,normal,4,cellmap,1.0)
v4 = Ahl*(Klop.lulop\Alh[4]) - Ahh4 + Ahh4bc
