using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=atol) for i = 1:length(v1)])
end

HL = reshape([0.0],1,1)
sfunc(x) = [0.5*x[1]*(x[1]+1)]
vfunc(x) = [(1.0-x[2])*(1.0+x[2])*x[1]*(x[1]-1.0)*0.5]
linemap = HDGElasticity.LineMap([-1.,-1.],[-1.,+1.])
squad = tensor_product_quadrature(1,3)
comp = [1.0]
D = reshape([1.0],1,1)

HDGElasticity.HLop!(HL,sfunc,vfunc,squad,linemap,comp,[D],1.,1)
@test allapprox(HL,[2/15])

fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,linemap,comp,[D],0.5,1)
@test allapprox(HL,0.5*[2/15])

comp = ones(length(squad))'
imap = InterpolatingPolynomial(2,1,1)
coeffs = [-1.0,-1.,-1.,1.]
update!(imap,coeffs)
fill!(HL,0.0)
scale = ones(length(squad))
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,imap,comp,comp,D,scale,1)
@test allapprox(HL,[2/15])

scale = 0.5*ones(length(squad))
fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,imap,comp,comp,D,scale,1)
@test allapprox(HL,0.5*[2/15])


sbasis = TensorProductBasis(1,1)
vbasis = TensorProductBasis(2,1)
linemap = HDGElasticity.LineMap([-1.,-1.],[-1.,1.])
comp = zeros(2,3)
components = [0.0,1.0]
normal = [-1.,0.0]
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.0,2.))
Ek = HDGElasticity.vec_to_symm_mat_converter(2)
NED = [normal[k]*Ek[k]'*Dhalf for k = 1:2]

cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
facejacobian = HDGElasticity.face_determinant_jacobian(cellmap)
HL = zeros(4,12)
HDGElasticity.HLop!(HL,sbasis,vbasis,squad,linemap,components,NED,facejacobian[4],cellmap)

HU = reshape([0.0],1,1)
comp = [1.0]
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,linemap,comp,1.,1)
@test allapprox(HU,[2/15])

fill!(HU,0.0)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,linemap,comp,0.5,1)
@test allapprox(HU,0.5*[2/15])

fill!(HU,0.0)
coeffs = [-1.,-1.,-1.,1.]
update!(imap,coeffs)
components = ones(length(squad))'
scale = ones(length(squad))
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,imap,components,scale,1)
@test allapprox(HU,[2/15])
