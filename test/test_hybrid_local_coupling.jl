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
vfunc(x) = [(1.0-x[1])*(1.0+x[1])]
squad = tensor_product_quadrature(1,3)
comp = reshape(ones(3),1,3)
D = reshape([1.0],1,1)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,comp,comp,D,D,1.,1,1,1,1)
@test allapprox(HL,[2/15])

fill!(HL,0.0)
HDGElasticity.HLop!(HL,sfunc,vfunc,squad,comp,comp,D,D,0.5,1,1,1,1)
@test allapprox(HL,0.5*[2/15])

sbasis = TensorProductBasis(1,1)
vbasis = TensorProductBasis(2,1)
comp = zeros(2,3)
comp[2,:] .= 1.0
normals = zeros(2,3)
normals[1,:] .= -1.0
Dhalf = diagm(ones(3))

cellmap = HDGElasticity.AffineMap([0.,0.],[2.,1.])
HL = HDGElasticity.HLop(sbasis,vbasis,squad,comp,normals,Dhalf,4,cellmap)
@test size(HL) == (4,12)

comp = [0.0,1.0]
normals = [-1.0,0.0]
HL = HDGElasticity.HLop(sbasis,vbasis,squad,comp,normals,Dhalf,4,cellmap)
@test size(HL) == (4,12)

HU = reshape([0.0],1,1)
comp = reshape(ones(3),1,3)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,comp,1.,1.,1,1,1)
@test allapprox(HU,[2/15])

fill!(HU,0.0)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,comp,0.5,1.,1,1,1)
@test allapprox(HU,0.5*[2/15])

fill!(HU,0.0)
HDGElasticity.HUop!(HU,sfunc,vfunc,squad,comp,1.,1/3,1,1,1)
@test allapprox(HU,1/3*[2/15])

comp = zeros(2,3)
comp[2,:] .= 1.0
HU = HDGElasticity.HUop(sbasis,vbasis,squad,comp,4,cellmap,1.)
@test size(HU) == (4,8)

comp = [0.0,1.0]
HU = HDGElasticity.HUop(sbasis,vbasis,squad,comp,4,cellmap,1.)
@test size(HU) == (4,8)

comp = [0.0,1.0]
normal = [-1.0,0.0]
Ahl = HDGElasticity.hybrid_local_operator(sbasis,vbasis,squad,comp,normal,
    Dhalf,4,cellmap,1.0)
@test size(Ahl) == (4,20)
