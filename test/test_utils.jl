using Test
using StaticArrays
using IntervalArithmetic
using PolynomialBasis
using ImplicitDomainQuadrature
using CartesianMesh
# using Revise
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

xL = @SVector [-1.,-1.,-1.,-1.]
xR = @SVector [+1.,+1.,+1.,+1.]
@test_throws AssertionError HDGE.AffineMap(xL,xR)
xL = @SVector [+1.,+1.]
xR = @SVector [-1.,-1.]
@test_throws AssertionError HDGE.AffineMap(xL,xR)

xL = @SVector [-1.,-1.]
xR = @SVector [+1.,+1.]
map = HDGE.AffineMap(xL,xR)
@test allequal(map.xL,xL)
@test allequal(map.xR,xR)
@test allapprox(HDGE.jacobian(map),[1.0,1.0])
@test allapprox(HDGE.inverse_jacobian(map),[1.0,1.0])
@test HDGE.determinant_jacobian(map) ≈ 1.0

xL = [-1.,-1.]
xR = [1.,1.,1.]
@test_throws AssertionError HDGE.AffineMap(xL,xR)
xR = [1.,1.]
map = HDGE.AffineMap(xL,xR)
@test allequal(map.xL,xL)
@test allequal(map.xR,xR)
@test_throws DimensionMismatch map([0.0,0.0,0.0])
@test allequal(map([0.0,0.0]),[0.0,0.0])
@test allequal(map([-1.0,0.0]),[-1.0,0.0])

xL = [0.,0.]
xR = [1.,1.]
map = HDGE.AffineMap(xL,xR)
@test allapprox(HDGE.jacobian(map),[0.5,0.5])
@test allapprox(HDGE.inverse_jacobian(map),[2.,2.])
@test HDGE.determinant_jacobian(map) ≈ 0.25
@test allequal(map([0.,0.]),[0.5,0.5])
@test allequal(map([-1.,0.]),[0.,0.5])
xi = [-0.5  -0.5  0.0  1.0
      -0.5  +0.0  0.5 -0.5]
x = map(xi)
testx = [0.25  0.25  0.50  1.0
         0.25  0.50  0.75  0.25]
@test allequal(x,testx)

xL = [0.,0.]
xR = [2.,1.]
map = HDGE.AffineMap(xL,xR)
@test allapprox(HDGE.jacobian(map),[1.0,0.5])
@test allapprox(HDGE.inverse_jacobian(map),[1.0,2.])
@test HDGE.determinant_jacobian(map) ≈ 0.5

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

f(x,y) = 2x + 3y
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
