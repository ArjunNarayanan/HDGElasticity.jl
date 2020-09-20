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

function Base.isequal(c1::HDGElasticity.CellMap,c2::HDGElasticity.CellMap)
    flag = true
    flag = flag && allapprox(c1.xiL,c2.xiL)
    flag = flag && allapprox(c1.xiR,c2.xiR)
    flag = flag && allapprox(c1.xL,c2.xL)
    flag = flag && allapprox(c1.xR,c2.xR)
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

@test HDGElasticity.is_interior_cell(connectivity[1]) == false
@test allequal(HDGElasticity.interior_cells(connectivity),[false,false])

basis = TensorProductBasis(2,1)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
cellsign = HDGElasticity.cell_signatures(coeffs,poly)
testcellsign = [0,1]
@test allequal(cellsign,testcellsign)

facemaps = HDGElasticity.reference_cell_facemaps(2)
cellmap = HDGElasticity.CellMap(domain[1])
facescale = HDGElasticity.face_determinant_jacobian(cellmap)
isinteriorcell = HDGElasticity.interior_cells(connectivity)
dgmesh = HDGElasticity.DGMesh(domain,connectivity,isinteriorcell,
    cellsign,facemaps,facescale,cellmap)
@test allequal(dgmesh.domain,domain)
@test all([allequal(dgmesh.connectivity[i],connectivity[i]) for i = 1:2])
@test allequal(dgmesh.cellsign,cellsign)
@test allequal(dgmesh.facescale,facescale)
@test isequal(dgmesh.cellmap,cellmap)

dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
@test allequal(dgmesh.domain,domain)
@test all([allequal(dgmesh.connectivity[i],connectivity[i]) for i = 1:2])
@test allequal(dgmesh.cellsign,cellsign)
@test allequal(dgmesh.facescale,facescale)
@test HDGElasticity.dimension(dgmesh) == 2
cellmap = HDGElasticity.CellMap(dgmesh.domain[1])
@test isequal(cellmap,dgmesh.cellmap)
