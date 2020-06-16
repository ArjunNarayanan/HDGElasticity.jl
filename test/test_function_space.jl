using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
using Revise
using HDGElasticity

function allequal(u,v)
    return all(u .== v)
end

function allapprox(u,v)
    return all(u .≈ v)
end

orders = [2 2 3 2
          2 1 4 2]
bs = HDGElasticity.bases(2,orders)
nf = (orders .+ 1).^2
nftest = HDGElasticity.number_of_basis_functions.(bs)
@test size(bs) == (2,4)
@test allequal(nf,nftest)

orders = reshape([rand(1:5) for i = 1:24],4,2,3)
bs = HDGElasticity.bases(1,orders)
nf = (orders .+ 1)
nftest = HDGElasticity.number_of_basis_functions.(bs)
@test size(bs) == (4,2,3)
@test allequal(nf,nftest)

bases = HDGElasticity.element_bases(2,3,5)
@test size(bases) == (2,5)
nftest = HDGElasticity.number_of_basis_functions.(bases)
@test all(nftest .== 16)

bases = HDGElasticity.hybrid_bases(1,4,5)
@test size(bases) == (4,2,5)
nf = HDGElasticity.number_of_basis_functions.(bases)
@test all(nf .== 5)

function distance_function(coords,xc)
    return coords[1,:] .- xc
end

x0 = [0.,0.]
widths = [2.,1.]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)
basis = TensorProductBasis(2,1)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
isactivecell = HDGElasticity.active_cells(coeffs,poly)
nqps = similar(isactivecell,Int)
fill!(nqps,2)
quads = HDGElasticity.element_quadratures(2,isactivecell,coeffs,poly,nqps)

quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
py = quad1d.points

p1,w1 = ImplicitDomainQuadrature.transform(quad1d,0.50,1.0)
testp1 = [p1[1]  p1[2]  p1[1]  p1[2]
          py[1]  py[1]  py[2]  py[2]]
testw1 = kron(quad1d.weights,w1)
@test allapprox(testp1,quads[1,1].points)
@test allapprox(testw1,quads[1,1].weights)

p2,w2 = ImplicitDomainQuadrature.transform(quad1d,-1.,0.5)
testp2 = [p2[1]  p2[2]  p2[1]  p2[2]
          py[1]  py[1]  py[2]  py[2]]
testw2 = kron(quad1d.weights,w2)
@test allapprox(testp2,quads[2,1].points)
@test allapprox(testw2,quads[2,1].weights)

qp3 = tensor_product_quadrature(2,2)
@test allequal(qp3.points,quads[1,2].points)
@test allequal(qp3.weights,quads[1,2].weights)

@test !isassigned(quads,2,2)
