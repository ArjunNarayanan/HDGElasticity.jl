using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
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

HDGElasticity.HLop!(HL,sfunc,vfunc,squad,linemap,comp,comp,[D],1.)
@test allapprox(HL,[2/15])

fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,linemap,comp,comp,[D],0.5)
@test allapprox(HL,0.5*[2/15])

fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,linemap,-comp,comp,[D],0.5)
@test allapprox(HL,-0.5*[2/15])

fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,linemap,comp,-comp,[D],0.5)
@test allapprox(HL,0.5*[2/15])

comp = ones(length(squad))'
imap = InterpolatingPolynomial(2,1,1)
coeffs = [-1.0,-1.,-1.,1.]
update!(imap,coeffs)
fill!(HL,0.0)
scale = ones(length(squad))
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,imap,comp,comp,D,scale)
@test allapprox(HL,[2/15])

scale = 0.5*ones(length(squad))
fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,imap,comp,comp,D,scale)
@test allapprox(HL,0.5*[2/15])


sbasis = TensorProductBasis(1,1)
vbasis = TensorProductBasis(2,1)
linemap = HDGElasticity.LineMap([-1.,-1.],[-1.,1.])
comp = zeros(2,3)
components = [0.0,1.0]
normal = [-1.,0.0]
Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.0,2.))
Ek = HDGElasticity.vec_to_symm_mat_converter(2)
ED = [Ek[k]'*Dhalf for k = 1:2]

cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
facejacobian = HDGElasticity.face_determinant_jacobian(cellmap)
HL = zeros(4,12)
HDGElasticity.HLop!(HL,sbasis,vbasis,squad,linemap,normal,components,ED,facejacobian[4])

normal = [-1.,0.]
components = [0.,1.]
facescale = HDGElasticity.face_determinant_jacobian(cellmap)
HLl = HDGElasticity.HLop(sbasis,vbasis,squad,linemap,normal,components,Dhalf,facescale[4])
@test size(HLl) == (4,12)

normal = repeat([-1.,0.],inner=(1,length(squad)))
components = repeat([0.,1.],inner=(1,length(squad)))
scale = repeat([facescale[4]],length(squad))
HLi = HDGElasticity.HLop(sbasis,vbasis,squad,imap,normal,components,Dhalf,scale)
@test size(HLi) == (4,12)
@test allapprox(HLl,HLi)

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

components = [0.,1.]
HUl = HDGElasticity.HUop(sbasis,vbasis,squad,linemap,components,facescale[4],2)
@test size(HUl) == (4,8)

components = repeat([0.,1.],inner=(1,length(squad)))
scale = repeat([facescale[4]],length(squad))
HUi = HDGElasticity.HUop(sbasis,vbasis,squad,imap,components,scale,2)
@test size(HUi) == (4,8)
@test allapprox(HUl,HUi)
