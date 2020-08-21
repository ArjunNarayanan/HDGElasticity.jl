using Test
using IntervalArithmetic
using PolynomialBasis
using CartesianMesh
# using Revise
using HDGElasticity

function allequal(v1,v2)
    return all(v1 .== v2)
end

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function distance_function(coords,xc)
    return coords[1,:] .- xc
end

x0 = [0.,0.]
widths = [2.,1.]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)

domain = HDGElasticity.cell_domain(mesh)
testdomain = [IntervalBox(0..1,0..1),IntervalBox(1..2,0..1)]
@test allequal(domain,testdomain)

connectivity = HDGElasticity.cell_connectivity(mesh)
testconn = [[(0,0),(2,4),(0,0),(0,0)],
            [(0,0),(0,0),(0,0),(1,2)]]
@test all([allequal(connectivity[i],testconn[i]) for i = 1:2])

basis = TensorProductBasis(2,1)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
cellsign = HDGElasticity.cell_signatures(coeffs,poly)
testcellsign = [0,1]
@test allequal(cellsign,testcellsign)

dgmesh = HDGElasticity.DGMesh(domain,connectivity,cellsign)
@test allequal(dgmesh.domain,domain)
@test all([allequal(dgmesh.connectivity[i],connectivity[i]) for i = 1:2])
@test allequal(dgmesh.cellsign,cellsign)

dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
@test allequal(dgmesh.domain,domain)
@test all([allequal(dgmesh.connectivity[i],connectivity[i]) for i = 1:2])
@test allequal(dgmesh.cellsign,cellsign)

@test HDGElasticity.dimension(dgmesh) == 2
