using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

sbasis = TensorProductBasis(1,1)
squad = tensor_product_quadrature(1,2)
facequads = repeat([squad],4)

cellmap = HDGElasticity.CellMap([0.,0.],[1.,0.5])
HH = HDGElasticity.HHop(sbasis,facequads,cellmap,1.)

testmatrix = [2/3 0.0 1/3 0.0
              0.0 2/3 0.0 1/3
              1/3 0.0 2/3 0.0
              0.0 1/3 0.0 2/3]

@test allapprox(HH[1],0.5*testmatrix)
@test allapprox(HH[2],0.25*testmatrix)
@test allapprox(HH[3],0.5*testmatrix)
@test allapprox(HH[4],0.25*testmatrix)

HH = HDGElasticity.HHop(sbasis,facequads,cellmap,2.)

@test allapprox(HH[1],testmatrix)
@test allapprox(HH[2],0.5*testmatrix)
@test allapprox(HH[3],testmatrix)
@test allapprox(HH[4],0.5*testmatrix)

HH = reshape([0.0],1,1)
sfunc(x) = [0.5*x[1]*(x[1]+1)]
squad = tensor_product_quadrature(1,3)
components = reshape(ones(3),1,3)

HDGElasticity.HHop!(HH,sfunc,squad,components,1.,1,1)
@test allapprox(HH,[4/15])

fill!(HH,0.0)
HDGElasticity.HHop!(HH,sfunc,squad,components,2.,1,1)
@test allapprox(HH,2*[4/15])

fill!(HH,0.0)
components = [1.0]
HDGElasticity.HHop!(HH,sfunc,squad,components,3.,1,1)
@test allapprox(HH,3*[4/15])

n = [1.,1.]/sqrt(2.)
normals = repeat(n,inner=(1,3))
imap = InterpolatingPolynomial(2,sbasis)
coeffs = [-0.5  -1.0
          -1.0  +0.0]
update!(imap,coeffs)
cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
HH = zeros(4,4)

HDGElasticity.HHop_on_interface!(HH,sbasis,squad,normals,imap,cellmap,1.)
@test allapprox(HH,0.5/sqrt(2.0)*testmatrix)

cellmap = HDGElasticity.CellMap([0.,0.],[1.,0.5])
coeffs = [1.0,-1.0,1.,1.]
update!(imap,coeffs)
normals = repeat([1.0,0.0],inner=(1,length(squad)))
HH = zeros(4,4)
HDGElasticity.HHop_on_interface!(HH,sbasis,squad,normals,normals,imap,cellmap,1.0)

testHH = 0.25*testmatrix
testHH[2,:] .= 0.0
testHH[4,:] .= 0.0
@test allapprox(HH,testHH)
