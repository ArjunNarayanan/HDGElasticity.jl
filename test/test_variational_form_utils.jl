using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using HDGElasticity

HDGE = HDGElasticity

function allequal(v1,v2)
    return all(v1 .== v2)
end

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,tol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
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

@test HDGE.symmetric_tensor_dimension(2) == 3
@test HDGE.symmetric_tensor_dimension(3) == 6

@test_throws ArgumentError HDGE.symmetric_tensor_dimension(1)
@test_throws ArgumentError HDGE.symmetric_tensor_dimension(4)


basis = TensorProductBasis(1,1)
quad = tensor_product_quadrature(1,2)
matrix = HDGE.mass_matrix(basis,quad,1,1.0)
testmatrix = [2/3 1/3
              1/3 2/3]
@test allapprox(matrix,testmatrix)

vals = [1. 1.]
rhs = HDGE.linear_form(vals,basis,quad)
@test allapprox(rhs,ones(2))

basis = TensorProductBasis(2,1)
facequad = tensor_product_quadrature(1,2)

map = InterpolatingPolynomial(2,1,1)
coeffs = [1.  1.
          -1. 1.]
update!(map,coeffs)
matrix = zeros(4,4)
HDGElasticity.update_mass_matrix!(matrix,basis,facequad,map,1,[1.,1.])
testmatrix = [0.0  0.0  0.0  0.0
           0.0  0.0  0.0  0.0
           0.0  0.0  2/3  1/3
           0.0  0.0  1/3  2/3]
@test allapprox(matrix,testmatrix)

matrix = zeros(4,4)
isactiveface = [true,false,false,true]
facequads = Vector{QuadratureRule{1}}(undef,4)
facequads[1] = facequad
facequads[4] = facequad
cellmap = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
HDGElasticity.update_mass_matrix_on_active_faces!(matrix,basis,facequads,isactiveface,1,cellmap)
testmatrix = [4/3  1/3  1/3  0.0
              1/3  2/3  0.0  0.0
              1/3  0.0  2/3  0.0
              0.0  0.0  0.0  0.0]
@test allapprox(matrix,testmatrix)


quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(2)
facequads = Vector{QuadratureRule{1}}(undef,4)
facequads[1] = QuadratureRule(ImplicitDomainQuadrature.transform(quad1d,-1.,0.)...)
facequads[4] = facequads[1]
matrix = zeros(4,4)
cellmap = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
HDGElasticity.update_mass_matrix_on_active_faces!(matrix,basis,
    facequads,isactiveface,1,cellmap)
testmatrix = [7/6  1/6  1/6  0.
              1/6  1/12 0.0  0.
              1/6  0.0  1/12 0.
              0.   0.   0.   0.]
@test allapprox(matrix,testmatrix)


N(x) = [(1.0-x[1])*(1.0-x[2])/4.0]
quad = tensor_product_quadrature(1,3)
imap = InterpolatingPolynomial(2,1,1)
icoeffs = [0.0  -1.0
          -1.0   0.0]
update!(imap,icoeffs)
matrix = reshape([0.0],1,1)
HDGElasticity.update_mass_matrix!(matrix,N,quad,imap,1,[1.,1.,1.])
@test matrix[1] ≈ 4.7/16*sqrt(2.)

basis = TensorProductBasis(2,1)
facequads = Vector{QuadratureRule{1}}(undef,4)
facequads[1] = QuadratureRule(ImplicitDomainQuadrature.transform(quad1d,-1.,0.)...)
facequads[4] = facequads[1]
iquad = tensor_product_quadrature(1,3)
isactiveface = [true,false,false,true]
cellmap = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
imap = InterpolatingPolynomial(2,1,1)
icoeffs = [0.0  -1.0
          -1.0   0.0]
update!(imap,icoeffs)
normals = reshape(1/sqrt(2)*ones(6),2,3)
matrix = HDGElasticity.mass_matrix_on_boundary(basis,facequads,isactiveface,
    iquad,normals,imap,1,cellmap)


facequad = tensor_product_quadrature(1,2)
matrix = HDGElasticity.mass_matrix_on_boundary(basis,facequad,1,cellmap)
testmatrix = [4/3 1/3 1/3 0.
              1/3 4/3 0. 1/3
              1/3 0. 4/3 1/3
              0.  1/3 1/3 4/3]
@test allapprox(matrix,testmatrix)
