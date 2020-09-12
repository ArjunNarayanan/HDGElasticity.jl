using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

function allapprox(v1,v2,tol)
    flag = length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
end

vbasis = TensorProductBasis(2,1)
sbasis = TensorProductBasis(1,1)
vquad = tensor_product_quadrature(2,2)
squad = tensor_product_quadrature(1,2)
facequads = repeat([squad],4)
facemaps = HDGElasticity.reference_cell_facemaps(2)
normals = HDGElasticity.reference_normals()
lambda = 1.0
mu = 2.0
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(lambda,mu))
cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
facescale = HDGElasticity.face_determinant_jacobian(cellmap)
stabilization = 0.1


locop = HDGElasticity.local_operator(vbasis,vquad,facequads,facemaps,
    Dhalf,stabilization,cellmap)
facelhops = HDGElasticity.local_hybrid_operator(vbasis,sbasis,facequads,facemaps,
    normals,Dhalf,stabilization,cellmap)
lhop = hcat(facelhops...)
HH = HDGElasticity.hybrid_operator(sbasis,facequads,stabilization,cellmap)
HHmass = HDGElasticity.hybrid_operator(sbasis,facequads,1.0,cellmap)

tcomp = [1.0,0.0]
hybloc1 = HDGElasticity.hybrid_local_operator_traction_components(sbasis,
    vbasis,facequads[1],facemaps[1],normals[1],tcomp,Dhalf,
    stabilization,facescale[1])
HHt1 = HDGElasticity.HHop(sbasis,squad,tcomp,stabilization,facescale[1])
ucomp = [0.0,-1.0]
HHu1 = HDGElasticity.HHop(sbasis,squad,ucomp,facescale[1])

hybloc2 = facelhops[2]'
R2 = -HDGElasticity.linear_form(facescale[2]*[1.0,0.0],sbasis,squad)
HH2 = HH[2]

hybloc3 = facelhops[3]'
HH3 = HH[3]

tcomp = [0.,1.]
hybloc4 = HDGElasticity.hybrid_local_operator_traction_components(sbasis,
    vbasis,facequads[4],facemaps[4],normals[4],tcomp,Dhalf,
    stabilization,facescale[4])
HHt4 = HDGElasticity.HHop(sbasis,squad,tcomp,stabilization*facescale[4])
ucomp = [-1.0,0.0]
HHu4 = HDGElasticity.HHop(sbasis,squad,ucomp,facescale[4])

hlop = vcat(hybloc1,hybloc2,hybloc3,hybloc4)

hhop = zeros(16,16)
hhop[1:4,1:4] = HHt1 - HHu1
hhop[5:8,5:8] = HH2
hhop[9:12,9:12] = HH3
hhop[13:16,13:16] = HHt4 - HHu4

cellop = hlop*(locop\lhop) - hhop
rhs = zeros(16)
rhs[5:8] = R2

sol = cellop\rhs
Hsol = reshape(sol,2,:)

locsol = locop\(lhop*sol)
L = -Dhalf*reshape(locsol[1:12],3,:)
U = reshape(locsol[13:20],2,:)

E = (lambda+2mu) - lambda^2/(lambda+2mu)
e11 = 1/E
e22 = -lambda/(lambda+2mu)*e11
u1 = e11*2
u2 = e22*1

testHsol = zeros(2,8)
testHsol[1,[2,3,4,6]] .= u1
testHsol[2,[4,5,6,8]] .= u2

testL = zeros(3,4)
testL[1,:] .= 1.0

testU = zeros(2,4)
testU[2,2] = u2
testU[1,3] = u1
testU[1,4] = u1
testU[2,4] = u2

@test allapprox(testL,L,1e-14)
@test allapprox(testU,U,1e-14)
@test allapprox(Hsol,testHsol,1e-14)



hybloc1 = zeros(4,20)
HH1 = HHmass[1]

hybloc2 = facelhops[2]'
R2 = -HDGElasticity.linear_form(facescale[2]*[0.0,1.0],sbasis,facequads[2])
HH2 = HH[2]

hybloc3 = facelhops[3]'
R3 = -HDGElasticity.linear_form(facescale[3]*[1.0,0.0],sbasis,facequads[3])
HH3 = HH[3]

hybloc4 = facelhops[4]'
R4 = -HDGElasticity.linear_form(facescale[4]*[0.0,-1.0],sbasis,facequads[4])
HH4 = HH[4]

hlop = vcat(hybloc1,hybloc2,hybloc3,hybloc4)

hhop = zeros(16,16)
hhop[1:4,1:4] = -HH1
hhop[5:8,5:8] = HH2
hhop[9:12,9:12] = HH3
hhop[13:16,13:16] = HH4

cellop = hlop*(locop\lhop) - hhop
rhs = zeros(16)
rhs[5:8] = R2
rhs[9:12] = R3
rhs[13:16] = R4

sol = cellop\rhs
Hsol = reshape(sol,2,:)

locsol = locop\(lhop*sol)
L = -Dhalf*reshape(locsol[1:12],3,:)
U = reshape(locsol[13:20],2,:)

e12 = 1.0/(2mu)
gamma = 2*e12

testL = zeros(3,4)
testL[3,:] .= 1.0

testU = zeros(2,4)
testU[1,2] = gamma
testU[1,4] = gamma

testH = zeros(2,8)
testH[1,4] = gamma
testH[1,5] = gamma
testH[1,6] = gamma
testH[1,8] = gamma

@test allapprox(Hsol,testH,1e-14)
@test allapprox(L,testL,1e-14)
@test allapprox(U,testU,1e-14)
