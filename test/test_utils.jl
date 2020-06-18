using Test
using StaticArrays
using IntervalArithmetic
using PolynomialBasis
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

basis = TensorProductBasis(2,4)
@test HDGElasticity.number_of_basis_functions(basis) == 25

@test_throws ArgumentError HDGE.reference_cell(3)
box = HDGE.reference_cell(1)
testbox = IntervalBox([-1.0],[1.0])
@test allequal(box,testbox)

box = HDGE.reference_cell(2)
testbox = IntervalBox([-1.,-1.],[1.,1.])
@test allequal(box,testbox)

normals = HDGE.reference_normals()
@test allequal(normals[1],[0.0,-1.0])
@test allequal(normals[2],[1.0,0.0])
@test allequal(normals[3],[0.0,1.0])
@test allequal(normals[4],[-1.0,0.0])

@test HDGE.reference_cell_volume(1) == 2.0
@test HDGE.reference_cell_volume(2) == 4.0
@test HDGE.reference_cell_volume(3) == 8.0

element_size = [1.0,2.0]
@test HDGE.affine_map_jacobian(element_size) == 0.5

m = [1.0 2.0
     3.0 4.0]
v = [5.0,6.0,7.0]
result = [5.0  10.0 6.0  12.0  7.0  14.0
          15.0 20.0 18.0 24.0  21.0 28.0]
@test allequal(HDGE.make_row_matrix(v,m),result)


result = [5.0 0.0 6.0 0.0 7.0 0.0
          0.0 5.0 0.0 6.0 0.0 7.0]
@test allequal(HDGE.interpolation_matrix(v,2),result)
result = [5.0 0.0 0.0 6.0 0.0 0.0 7.0 0.0 0.0
          0.0 5.0 0.0 0.0 6.0 0.0 0.0 7.0 0.0
          0.0 0.0 5.0 0.0 0.0 6.0 0.0 0.0 7.0]
@test allequal(HDGE.interpolation_matrix(v,3),result)


Ek = HDGE.vec_to_symm_mat_converter(2)
@test length(Ek) == 2
E1 = [1.0 0.0 0.0
      0.0 0.0 1.0]'
E2 = [0.0 0.0 1.0
      0.0 1.0 0.0]'
@test allequal(Ek[1],E1)
@test allequal(Ek[2],E2)

Ek = HDGElasticity.vec_to_symm_mat_converter(3)
@test length(Ek) == 3
E1 = zeros(3,6)
E1[1,1] = 1.0
E1[2,4] = 1.0
E1[3,5] = 1.0
E1 = E1'
@test allequal(Ek[1],E1)
E2 = zeros(3,6)
E2[1,4] = 1.0
E2[2,2] = 1.0
E2[3,6] = 1.0
E2 = E2'
@test allequal(Ek[2],E2)
E3 = zeros(3,6)
E3[1,5] = 1.0
E3[2,6] = 1.0
E3[3,3] = 1.0
E3 = E3'
@test allequal(Ek[3],E3)

@test_throws ArgumentError HDGE.vec_to_symm_mat_converter(4)
@test_throws ArgumentError HDGE.vec_to_symm_mat_converter(1)

@test HDGE.symmetric_tensor_dim(2) == 3
@test HDGE.symmetric_tensor_dim(3) == 6

@test_throws ArgumentError HDGE.symmetric_tensor_dim(1)
@test_throws ArgumentError HDGE.symmetric_tensor_dim(4)

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
@test allequal(map([0.,0.]),[0.5,0.5])
@test allequal(map([-1.,0.]),[0.,0.5])
xi = [-0.5  -0.5  0.0  1.0
      -0.5  +0.0  0.5 -0.5]
x = map(xi)
testx = [0.25  0.25  0.50  1.0
         0.25  0.50  0.75  0.25]
@test allequal(x,testx)

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

poly = InterpolatingPolynomial(1,1,3)
@test HDGE.dimension(poly) == 1
poly = InterpolatingPolynomial(1,2,3)
@test HDGElasticity.dimension(poly) == 2
