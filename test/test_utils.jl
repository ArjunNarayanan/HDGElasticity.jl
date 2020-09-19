using Test
using LinearAlgebra
using IntervalArithmetic
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
using HDGElasticity

HDGE = HDGElasticity

function allequal(v1,v2)
    return all(v1 .== v2)
end

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

@test HDGE.default_integer_type() == typeof(1)
@test HDGE.default_float_type() == typeof(1.0)

@test HDGE.reference_element_length() == 2.0

@test HDGElasticity.in_reference_interval(-0.25)
@test !HDGElasticity.in_reference_interval(+1.2)

@test HDGElasticity.reference_interval_1d() == (-1.0,+1.0)

@test_throws ArgumentError HDGE.reference_cell(3)
box = HDGE.reference_cell(1)
testbox = IntervalBox([-1.0],[1.0])
@test allequal(box,testbox)

box = HDGE.reference_cell(2)
testbox = IntervalBox([-1.,-1.],[1.,1.])
@test allequal(box,testbox)

@test_throws ArgumentError HDGElasticity.number_of_faces(1)
@test_throws ArgumentError HDGElasticity.number_of_faces(4)
@test HDGElasticity.number_of_faces(2) == 4
@test HDGElasticity.number_of_faces(3) == 6

normals = HDGE.reference_normals()
@test allequal(normals[1],[0.0,-1.0])
@test allequal(normals[2],[1.0,0.0])
@test allequal(normals[3],[0.0,1.0])
@test allequal(normals[4],[-1.0,0.0])

@test HDGE.reference_cell_volume(1) == 2.0
@test HDGE.reference_cell_volume(2) == 4.0
@test HDGE.reference_cell_volume(3) == 8.0




map = InterpolatingPolynomial(2,1,1)
coeffs = [0.0  0.0
          0.0  1.0]
update!(map,coeffs)
@test HDGE.determinant_jacobian(map,0.0) ≈ 0.5

map = InterpolatingPolynomial(2,1,2)
coeffs = [-1. 0. 1.
           0. -1 0]
update!(map,coeffs)
@test HDGE.determinant_jacobian(map,-0.5) ≈ sqrt(2)

mesh = UniformMesh([0.0,0.0],[1.,1.],[1,1])
basis = TensorProductBasis(2,2)
coords = HDGElasticity.nodal_coordinates(mesh,basis)
testcoords = [0.0 0.0 0.0 0.5 0.5 0.5 1.0 1.0 1.0
              0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0]
@test allequal(coords,testcoords)

@test HDGElasticity.neighbor_faceid(1) == 3
@test HDGElasticity.neighbor_faceid(2) == 4
@test HDGElasticity.neighbor_faceid(3) == 1
@test HDGElasticity.neighbor_faceid(4) == 2
@test_throws ArgumentError HDGElasticity.neighbor_faceid(5)
@test_throws ArgumentError HDGElasticity.neighbor_faceid(0)

f(x) = 2x[1] + 3x[2]
box = HDGE.reference_cell(2)
restricted_funcs = HDGE.restrict_on_faces(f,box)
@test allequal(restricted_funcs[1].([-0.5,0.5]),[-4.,-2.0])
@test allequal(restricted_funcs[2].([-0.5,0.5]),[0.5,3.5])
@test allequal(restricted_funcs[3].([-0.5,0.5]),[2.0,4.0])
@test allequal(restricted_funcs[4].([-0.5,0.5]),[-3.5,-0.5])


basis = TensorProductBasis(2,4)
@test HDGElasticity.number_of_basis_functions(basis) == 25
@test HDGElasticity.dimension(basis) == 2
@test HDGElasticity.number_of_basis_functions(basis.basis) == 5

poly = InterpolatingPolynomial(1,1,3)
@test HDGE.dimension(poly) == 1
poly = InterpolatingPolynomial(1,2,3)
@test HDGElasticity.dimension(poly) == 2

quad = tensor_product_quadrature(1,5)
@test HDGElasticity.number_of_quadrature_points(quad) == 5
@test HDGElasticity.dimension(quad) == 1
quad = tensor_product_quadrature(2,6)
@test HDGElasticity.dimension(quad) == 2
@test HDGElasticity.number_of_quadrature_points(quad) == 36


function plane_distance_function(coords,n,x0)
    return [n'*(coords[:,idx]-x0) for idx in 1:size(coords)[2]]
end
poly = InterpolatingPolynomial(1,2,1)
coords = [0. 0. 2.  2.
          0. 1. 0.  1.]
testn = [1.,1.]/sqrt(2.)
coeffs = plane_distance_function(coords,testn,[0.5,0.])
update!(poly,coeffs)
cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
p = [-0.25,-0.5]
n = HDGElasticity.levelset_normal(poly,p,cellmap)
@test allapprox(testn,n)

levelsetcoeffs = HDGElasticity.levelset_coefficients(
    x->plane_distance_function(x,[1.,0.],[0.5,0.]),mesh,poly.basis)
testlevelsetcoeffs = [-0.5,-0.5,0.5,0.5]
@test allapprox(levelsetcoeffs,testlevelsetcoeffs)

points = [-0.4  -0.9
          -0.8   -0.2]
n = HDGElasticity.levelset_normal(poly,points,cellmap)
@test allapprox(n,hcat(testn,testn))


normal = [1,2]
@test HDGElasticity.tangents(normal) == [2,-1]

normals = [1 2 3 4
           5 6 7 8]
testt = [5 6 7 8
         -1 -2 -3 -4]
@test allapprox(HDGElasticity.tangents(normals),testt)

cellmap = HDGElasticity.CellMap([0.,0.],[2.,1.])
@test HDGElasticity.scale_area(cellmap,[1.,0.]) ≈ 0.5
@test HDGElasticity.scale_area(cellmap,[0.,1.]) ≈ 1.0
n = [1. 0.
     0. 1.]
@test allapprox(HDGElasticity.scale_area(cellmap,n),[0.5,1.])


facequads = [[],[1,2,3],[1,2,3,4,5],[]]
isactiveface = HDGElasticity.active_faces(facequads)
@test allequal(isactiveface,[0,1,1,0])

cellmap = HDGElasticity.CellMap([3.,2.],[5.,7.])
midpoints = HDGElasticity.face_midpoints(cellmap)
@test length(midpoints) == 4
@test allapprox(midpoints[1],[4.,2.])
@test allapprox(midpoints[2],[5.,4.5])
@test allapprox(midpoints[3],[4.,7.])
@test allapprox(midpoints[4],[3.,4.5])

imap = InterpolatingPolynomial(2,1,2)
coeffs = [-1.,0.,0.,1.,1.,0.]
update!(imap,coeffs)
p = [-0.5,0.5]
normals = HDGElasticity.curve_normals(imap,p)
testn = [-1. 1.
          1. 1.]
testn .*= 1.0/sqrt(2.)
@test allapprox(testn,normals)
