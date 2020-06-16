using Test
using IntervalArithmetic
using PolynomialBasis
using CartesianMesh
using Revise
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
testconn = [0  0
            2  0
            0  0
            0  1]
@test allequal(connectivity,testconn)

basis = TensorProductBasis(2,1)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
isactivecell = HDGElasticity.active_cells(coeffs,poly)
testactivecells = [true true
                   true false]
@test allequal(isactivecell,testactivecells)

isactiveface = HDGElasticity.active_faces(coeffs,poly,isactivecell)
testactiveface = zeros(Bool,4,2,2)
testactiveface[:,1,1] = [true,true,true,false]
testactiveface[:,1,2] = [true,true,true,true]
testactiveface[:,2,1] = [true,false,true,true]
testactiveface[:,2,2] = [false,false,false,false]
@test allequal(isactiveface,testactiveface)
@test_throws ArgumentError HDGElasticity.update_active_faces!(isactiveface,2,2,2)

cell2elid = HDGElasticity.number_elements(isactivecell)
testcell2elid = [1 3
                 2 0]
@test allequal(cell2elid,testcell2elid)

@test HDGElasticity.neighbor_faceid(1) == 3
@test HDGElasticity.neighbor_faceid(2) == 4
@test HDGElasticity.neighbor_faceid(3) == 1
@test HDGElasticity.neighbor_faceid(4) == 2
@test_throws ArgumentError HDGElasticity.neighbor_faceid(0)
@test_throws ArgumentError HDGElasticity.neighbor_faceid(5)

face2hid = HDGElasticity.number_face_hybrid_elements(isactiveface,connectivity)
testface2hid = zeros(Int,4,2,2)
testface2hid[1,1,1] = 1
testface2hid[2,1,1] = 2
testface2hid[3,1,1] = 3

testface2hid[1,2,1] = 4
testface2hid[3,2,1] = 5
testface2hid[4,2,1] = 6

testface2hid[1,1,2] = 7
testface2hid[2,1,2] = 8
testface2hid[3,1,2] = 9
testface2hid[4,1,2] = 2
@test allequal(face2hid,testface2hid)

hid = maximum(face2hid)+1
interface2hid = HDGElasticity.number_interface_hybrid_elements(isactivecell,hid)
testinterface2hid = [10,0]
@test allequal(interface2hid,testinterface2hid)

dgmesh = HDGElasticity.DGMesh(domain,connectivity,isactivecell,
    isactiveface,cell2elid,face2hid,interface2hid)
@test allequal(dgmesh.domain,domain)
@test allequal(dgmesh.connectivity,connectivity)
@test allequal(dgmesh.isactivecell,isactivecell)
@test allequal(dgmesh.isactiveface,isactiveface)
@test allequal(dgmesh.cell2elid,cell2elid)
@test allequal(dgmesh.face2hid,face2hid)
@test allequal(dgmesh.interface2hid,interface2hid)

dgmesh = HDGElasticity.DGMesh(mesh,coeffs,poly)
@test allequal(dgmesh.domain,domain)
@test allequal(dgmesh.connectivity,connectivity)
@test allequal(dgmesh.isactivecell,isactivecell)
@test allequal(dgmesh.isactiveface,isactiveface)
@test allequal(dgmesh.cell2elid,cell2elid)
@test allequal(dgmesh.face2hid,face2hid)
@test allequal(dgmesh.interface2hid,interface2hid)
