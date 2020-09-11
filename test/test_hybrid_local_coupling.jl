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
icomp = ones(length(squad))'
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

imap = InterpolatingPolynomial(2,1,1)
coeffs = [-1.0,-1.,-1.,1.]
update!(imap,coeffs)
fill!(HL,0.0)
scale = ones(length(squad))
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,imap,icomp,icomp,D,scale)
@test allapprox(HL,[2/15])

fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,imap,icomp,icomp,D,0.5*scale)
@test allapprox(HL,0.5*[2/15])


sbasis = TensorProductBasis(1,1)
vbasis = TensorProductBasis(2,1)
linemap = HDGElasticity.LineMap([-1.,-1.],[-1.,1.])
components = [0.0,1.0]
normals = [-1.,0.0]
icomponents = repeat(components,inner=(1,length(squad)))
inormals = repeat(normals,inner=(1,length(squad)))

Dhalf = sqrt(HDGElasticity.plane_strain_voigt_hooke_matrix_2d(1.0,2.))
Ek = HDGElasticity.vec_to_symm_mat_converter(2)
ED = [Ek[k]'*Dhalf for k = 1:2]

cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
facejacobian = HDGElasticity.face_determinant_jacobian(cellmap)
HL = zeros(4,12)
HDGElasticity.HLop!(HL,sbasis,vbasis,squad,linemap,normals,components,ED,facejacobian[4])

HLl = HDGElasticity.HLop(sbasis,vbasis,squad,linemap,normals,components,Dhalf,facejacobian[4])
@test size(HLl) == (4,12)

iscale = HDGElasticity.scale_area(cellmap,inormals)
HLi = HDGElasticity.HLop(sbasis,vbasis,squad,imap,inormals,icomponents,Dhalf,iscale)
@test size(HLi) == (4,12)
@test allapprox(HLl,HLi)

HLl2 = HDGElasticity.HLop(sbasis,vbasis,squad,linemap,normals,components,Dhalf,0.5facejacobian[4])
@test allapprox(0.5HLl,HLl2)

HU = reshape([0.0],1,1)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,linemap,comp,1.,1)
@test allapprox(HU,[2/15])

fill!(HU,0.0)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,linemap,comp,0.5,1)
@test allapprox(HU,0.5*[2/15])

fill!(HU,0.0)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,linemap,-comp,0.5,1)
@test allapprox(HU,0.5*[2/15])

fill!(HU,0.0)
coeffs = [-1.,-1.,-1.,1.]
update!(imap,coeffs)
scale = ones(length(squad))
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,imap,icomp,scale,1)
@test allapprox(HU,[2/15])

fill!(HU,0.0)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,imap,icomp,0.5scale,1)
@test allapprox(HU,0.5*[2/15])


HUl = HDGElasticity.HUop(sbasis,vbasis,squad,linemap,components,facejacobian[4])
@test size(HUl) == (4,8)

scale = HDGElasticity.scale_area(cellmap,inormals)
HUi = HDGElasticity.HUop(sbasis,vbasis,squad,imap,icomponents,scale)
@test size(HUi) == (4,8)
@test allapprox(HUl,HUi)

HUl2 = HDGElasticity.HUop(sbasis,vbasis,squad,linemap,components,2*facejacobian[4])
@test size(HUl) == (4,8)
@test allapprox(2HUl,HUl2)

hybloc1 = HDGElasticity.hybrid_local_operator_traction_components(sbasis,vbasis,
    squad,linemap,normals,components,Dhalf,1.,facejacobian[4])
@test size(hybloc1) == (4,20)

hybloc2 = HDGElasticity.hybrid_local_operator_traction_components(sbasis,vbasis,
    squad,linemap,normals,components,Dhalf,0.1,facejacobian[4])
@test allapprox(0.1hybloc1,hybloc2)


iscale = HDGElasticity.scale_area(cellmap,inormals)
hybloc1 = HDGElasticity.hybrid_local_operator_traction_components(sbasis,vbasis,
    squad,imap,inormals,icomponents,Dhalf,1.,iscale)
@test size(hybloc1) == (4,20)

hybloc2 = HDGElasticity.hybrid_local_operator_traction_components(sbasis,vbasis,
    squad,imap,inormals,icomponents,Dhalf,0.2,iscale)
@test allapprox(0.2hybloc1,hybloc2)
