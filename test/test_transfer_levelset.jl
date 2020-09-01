using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,tol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
end

levelset = InterpolatingPolynomial(1,2,1)
coeffs = [-1.0,-1.0,1.0,1.0]
coords = [0.1 0.2 0.3 0.4
          0.1 0.2 0.3 0.4]
update!(levelset,coeffs)
vals = HDGElasticity.evaluate_levelset(levelset,coords)
testvals = coords[1,:]
@test allapprox(vals,testvals)

coeffs = [-1.0 0.0
          -1.0 0.0
          +1.0 2.0
          +1.0 2.0]
vals = HDGElasticity.evaluate_multiple_levelsets(levelset,coeffs,coords)
@test length(vals) == 2
t1 = coords[1,:]
t2 = [1.1,1.2,1.3,1.4]
@test allapprox(vals[1],t1)
@test allapprox(vals[2],t2)

coeffs = [0.0 1.0
          0.0 1.0
          1.0 2.0
          1.0 2.0]

cellmaps = [HDGElasticity.CellMap([-1.0,-1.0],[0.0,1.0]),
            HDGElasticity.CellMap([0.0,-1.0],[1.0,1.0])]
basis = levelset.basis
quad = tensor_product_quadrature(2,2)
mass_matrix = HDGElasticity.mass_matrix(basis,quad,1.0,1)
newcoeffs = HDGElasticity.transfer_levelset(levelset,coeffs,cellmaps,basis,quad,mass_matrix,4)
testnewcoeffs = [0.0,0.0,2.0,2.0]
@test allapprox(newcoeffs,testnewcoeffs,1e-15)


quadratic(x,y) = x^2 + 2x*y

cellmaps = [HDGElasticity.CellMap([-1.0,-1.0],[0.0,1.0]),
            HDGElasticity.CellMap([0.0,-1.0],[1.0,1.0])]

levelset = InterpolatingPolynomial(1,2,2)
basis = levelset.basis
quad = tensor_product_quadrature(2,3)
mass_matrix = HDGElasticity.mass_matrix(basis,quad,1.0,1)
coords1 = cellmaps[1](basis.points)
coords2 = cellmaps[2](basis.points)
cf1 = [quadratic(coords1[:,i]...) for i = 1:9]
cf2 = [quadratic(coords2[:,i]...) for i = 1:9]
coeffs = [cf1 cf2]
newcoeffs = HDGElasticity.transfer_levelset(levelset,coeffs,cellmaps,basis,quad,mass_matrix)

coords = basis.points
testnewcoeffs = [quadratic(coords[:,i]...) for i = 1:9]
@test allapprox(newcoeffs,testnewcoeffs,1e-14)
