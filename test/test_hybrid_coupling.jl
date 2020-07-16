using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

sbasis = TensorProductBasis(1,1)
squad = tensor_product_quadrature(1,2)

cellmap = HDGElasticity.AffineMap([0.,0.],[1.,0.5])

HH = HDGElasticity.HHop(sbasis,squad,cellmap,1.)

testmatrix = [2/3 0.0 1/3 0.0
              0.0 2/3 0.0 1/3
              1/3 0.0 2/3 0.0
              0.0 1/3 0.0 2/3]

@test allapprox(HH[1],0.5*testmatrix)
@test allapprox(HH[2],0.25*testmatrix)
@test allapprox(HH[3],0.5*testmatrix)
@test allapprox(HH[4],0.25*testmatrix)

HH = HDGElasticity.HHop(sbasis,squad,cellmap,2.)

testmatrix = [2/3 0.0 1/3 0.0
              0.0 2/3 0.0 1/3
              1/3 0.0 2/3 0.0
              0.0 1/3 0.0 2/3]

@test allapprox(HH[1],testmatrix)
@test allapprox(HH[2],0.5*testmatrix)
@test allapprox(HH[3],testmatrix)
@test allapprox(HH[4],0.5*testmatrix)

quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
tq = QuadratureRule(quad1d.points,quad1d.weights)
fq = QuadratureRule(ImplicitDomainQuadrature.transform(quad1d,-1.,0.)...)

facequads = [fq,tq,fq,tq]
isactiveface = [true,false,true,true]

HH = HDGElasticity.HHop_on_active_faces(sbasis,facequads,isactiveface,cellmap,1.)

cuttestmatrix = [7/12 0.0  1/6  0.0
                 0.0  7/12 0.0  1/6
                 1/6  0.0  1/12 0.0
                 0.0  1/6  0.0  1/12]

@test allapprox(HH[1],0.5*cuttestmatrix)
@test allapprox(HH[2],zeros(4,4))
@test allapprox(HH[3],0.5*cuttestmatrix)
@test allapprox(HH[4],0.25*testmatrix)

n = [1.,1.]/sqrt(2.)
normals = repeat(n,inner=(1,2))
imap = InterpolatingPolynomial(2,sbasis)
coeffs = [-0.5  -1.0
          -1.0  +0.0]
update!(imap,coeffs)
cellmap = HDGElasticity.AffineMap([0.,0.],[2.,1.])

HH = HDGElasticity.HHop_on_interface(sbasis,squad,normals,imap,cellmap,1.)
@test allapprox(HH,0.5/sqrt(2.0)*testmatrix)
