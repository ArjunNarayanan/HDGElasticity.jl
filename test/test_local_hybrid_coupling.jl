using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,atol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=atol) for i = 1:length(v1)])
end

vbasis = TensorProductBasis(2,1)
sbasis = TensorProductBasis(1,1)
squad = tensor_product_quadrature(1,2)

LH = zeros(4,2)
M = [1.0]
HDGElasticity.update_LHop!(LH,x->vbasis(extend(x,2,-1.0)),sbasis,squad,M,1.0,1)
LHtest =  [2/3  1/3
           0.0  0.0
           1/3  2/3
           0.0  0.0]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
testjac = 3.0
HDGElasticity.update_LHop!(LH,x->vbasis(extend(x,1,1.0)),sbasis,squad,M,testjac,1)
LHtest =  testjac*[0.0  0.0
                   0.0  0.0
                   2/3  1/3
                   1/3  2/3]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
HDGElasticity.update_LHop!(LH,x->vbasis(extend(x,2,1.0)),sbasis,squad,M,+1.0,1)
LHtest = [0.0   0.0
          +2/3  +1/3
           0.0   0.0
          +1/3  +2/3]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
HDGElasticity.update_LHop!(LH,x->vbasis(extend(x,1,-1.0)),sbasis,squad,M,+1.0,1)
LHtest =  [+2/3  +1/3
           +1/3  +2/3
            0.0   0.0
            0.0   0.0]
@test allapprox(LH,LHtest)


vf(x) = [x[1] + x[2]]
sf(x) = [x]
squad = tensor_product_quadrature(1,4)
nk = ones(length(squad))
imap = InterpolatingPolynomial(2,sbasis)
update!(imap,[0.,-1.,-1.,0.])
LH = reshape([0.0],1,1)
HDGElasticity.update_LHop!(LH,vf,sf,squad,nk,[1.0],imap,1,nk)
@test allapprox(LH,[0.0],1e-15)

normals = repeat([1.,1.]/sqrt(2.),inner=(1,length(squad)))
Dhalf = diagm(ones(3))
cellmap = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
LH = HDGElasticity.LHop_on_interface(vbasis,sbasis,squad,normals,Dhalf,imap,cellmap)
@test size(LH) == (12,4)

fq = tensor_product_quadrature(1,2)
LH = HDGElasticity.LHop(vbasis,sbasis,fq,Dhalf,cellmap)
@test all([size(i) == (12,4) for i in LH])

quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
facequads = Vector{QuadratureRule{1}}(undef,4)
fq = QuadratureRule(ImplicitDomainQuadrature.transform(quad1d,-1.,0.)...)
facequads[1] = fq
facequads[4] = fq
isactiveface = [true,false,false,true]
cellmap = HDGElasticity.AffineMap([0.,0.],[1.,2.])
Dhalf = diagm(ones(3))
LH = HDGElasticity.LHop_on_active_faces(vbasis,sbasis,facequads,isactiveface,Dhalf,cellmap)
s = size.(LH)
@test all(i == (12,4) for i in s)
@test all(LH[2] .≈ 0.0)
@test all(LH[3] .≈ 0.0)

UH = zeros(4,2)
HDGElasticity.UHop!(UH,x->vbasis(extend(x,2,-1.0)),sbasis,squad,1.0,1.0,1,4,2)
UHtest = [2/3  1/3
           0.0  0.0
           1/3  2/3
           0.0  0.0]
@test allapprox(UH,UHtest)

UH = zeros(4,2)
testjac = 2.0
HDGElasticity.UHop!(UH,x->vbasis(extend(x,1,+1.0)),
    sbasis,squad,1.0,testjac,1,4,2)
UHtest = testjac*[0.0  0.0
                   0.0  0.0
                   2/3  1/3
                   1/3  2/3]
@test allapprox(UH,UHtest)


UH = zeros(4,2)
HDGElasticity.UHop!(UH,x->vbasis(extend(x,2,+1.0)),
    sbasis,squad,1.0,+1.0,1,4,2)
UHtest = [0.0   0.0
          +2/3  +1/3
           0.0   0.0
          +1/3  +2/3]
@test all(UH .≈ UHtest)

UH = zeros(4,2)
HDGElasticity.UHop!(UH,x->vbasis(extend(x,1,-1.0)),
    sbasis,squad,1.0,+1.0,1,4,2)
UHtest = [+2/3  +1/3
           +1/3  +2/3
            0.0   0.0
            0.0   0.0]
@test all(UH .≈ UHtest)

UH = HDGElasticity.UHop(vbasis,sbasis,squad,cellmap,1.)
@test all([size(i) == (8,4) for i in UH])

vf(x) = [x[1] + x[2]]
sf(x) = [x]
squad = tensor_product_quadrature(1,4)
scale = ones(length(squad))
imap = InterpolatingPolynomial(2,1,1)
update!(imap,[0.,-1.,-1.,0.])
UH = reshape([0.0],1,1)
HDGElasticity.UHop_on_interface!(UH,vf,sf,squad,1.,imap,1,1,1,scale)
@test allapprox(UH,[0.0],1e-15)

cellmap = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
normals = reshape(1/sqrt(2)*ones(2*length(squad)),2,:)
UH = HDGElasticity.UHop_on_interface(vbasis,sbasis,squad,normals,imap,cellmap,1.)
@test size(UH) == (8,4)

quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
facequads = Vector{QuadratureRule{1}}(undef,4)
fq = QuadratureRule(ImplicitDomainQuadrature.transform(quad1d,-1.,0.)...)
facequads[1] = fq
facequads[4] = fq
isactiveface = [true,false,false,true]
cellmap = HDGElasticity.AffineMap([0.,0.],[1.,2.])
UH = HDGElasticity.UHop_on_active_faces(vbasis,sbasis,facequads,
    isactiveface,cellmap,1.)
s = size.(UH)
@test all([i == (8,4) for i in s])

lhc = HDGElasticity.local_hybrid_operator(vbasis,sbasis,squad,Dhalf,cellmap,1.)
@test all([size(i) == (20,4) for i in lhc])

lhc = HDGElasticity.local_hybrid_operator_on_active_faces(vbasis,sbasis,facequads,
    isactiveface,Dhalf,cellmap,1.)
@test all([size(i) == (20,4) for i in lhc])


lhc = HDGElasticity.local_hybrid_operator_on_interface(vbasis,sbasis,squad,
    normals,Dhalf,imap,cellmap,1.)
@test size(lhc) == (20,4)
