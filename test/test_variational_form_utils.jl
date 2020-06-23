using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

HDGE = HDGElasticity

function allequal(v1,v2)
    return all(v1 .== v2)
end

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

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


basis = TensorProductBasis(1,1)
quad = tensor_product_quadrature(1,2)
matrix = HDGE.mass_matrix(1,basis,quad)
testmatrix = [2/3 1/3
              1/3 2/3]
@test allapprox(matrix,testmatrix)

vals = [1. 1.]
rhs = HDGE.linear_form(vals,basis,quad)
@test allapprox(rhs,ones(2))
