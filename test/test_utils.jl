using Test, StaticArrays, LinearAlgebra
using CartesianMesh
using ImplicitDomainQuadrature
using HDGElasticity

@test HDGElasticity.get_reference_element_size(ReferenceQuadratureRule) ≈ 2.0

basis = TensorProductBasis(2,4)
@test HDGElasticity.number_of_basis_functions(basis) == 25

x0,dx = HDGElasticity.reference_element(basis)
@test all(x0 .≈ [-1.0,-1.0])
@test all(dx .≈ [2.0,2.0])

normals = HDGElasticity.reference_normals(basis)
@test all(normals[1] .≈ [0.0,-1.0])
@test all(normals[2] .≈ [1.0,0.0])
@test all(normals[3] .≈ [0.0,1.0])
@test all(normals[4] .≈ [-1.0,0.0])

element_size = @SVector [1.0,2.0]
quadrature_weights = @SVector [0.5,1.0]
@test HDGElasticity.jacobian(element_size,quadrature_weights) == 2.0/1.5

mesh = UniformMesh([1.0,2.0],[2.0,1.0],[10,3])
quad = TensorProductQuadratureRule(2,4)
@test HDGElasticity.jacobian(mesh,quad) ≈ 2.0/(30)/4.0

m = [1.0 2.0
     3.0 4.0]
v = [5.0,6.0,7.0]
result = [5.0  10.0 6.0  12.0  7.0  14.0
          15.0 20.0 18.0 24.0  21.0 28.0]
@test all(HDGElasticity.make_row_matrix(v,m) .== result)

result = [5.0 0.0 6.0 0.0 7.0 0.0
          0.0 5.0 0.0 6.0 0.0 7.0]
@test all(HDGElasticity.interpolation_matrix(v,2) .== result)
result = [5.0 0.0 0.0 6.0 0.0 0.0 7.0 0.0 0.0
          0.0 5.0 0.0 0.0 6.0 0.0 0.0 7.0 0.0
          0.0 0.0 5.0 0.0 0.0 6.0 0.0 0.0 7.0]
@test all(HDGElasticity.interpolation_matrix(v,3) .== result)


Ek = HDGElasticity.vec_to_symm_mat_converter(2)
@test length(Ek) == 2
E1 = [1.0 0.0 0.0
      0.0 0.0 1.0]'
E2 = [0.0 0.0 1.0
      0.0 1.0 0.0]'
@test all(Ek[1] .== E1)
@test all(Ek[2] .== E2)

Ek = HDGElasticity.vec_to_symm_mat_converter(3)
@test length(Ek) == 3
E1 = zeros(3,6)
E1[1,1] = 1.0
E1[2,4] = 1.0
E1[3,5] = 1.0
E1 = E1'
@test all(Ek[1] .== E1)
E2 = zeros(3,6)
E2[1,4] = 1.0
E2[2,2] = 1.0
E2[3,6] = 1.0
E2 = E2'
@test all(Ek[2] .== E2)
E3 = zeros(3,6)
E3[1,5] = 1.0
E3[2,6] = 1.0
E3[3,3] = 1.0
E3 = E3'
@test all(Ek[3] .== E3)

@test_throws ArgumentError HDGElasticity.vec_to_symm_mat_converter(4)
@test_throws ArgumentError HDGElasticity.vec_to_symm_mat_converter(1)

@test HDGElasticity.symmetric_tensor_dim(2) == 3
@test HDGElasticity.symmetric_tensor_dim(3) == 6

@test_throws ArgumentError HDGElasticity.symmetric_tensor_dim(1)
@test_throws ArgumentError HDGElasticity.symmetric_tensor_dim(4)

element_size = @SVector [0.5,0.1]
jacobian = HDGElasticity.AffineMapJacobian(element_size,2.0)
@test all(jacobian.jac .== [0.25,0.05])
@test all(jacobian.invjac .== [4.0,20.0])
@test jacobian.detjac == 0.25*0.05

jacobian = HDGElasticity.AffineMapJacobian(element_size,quad)
@test all(jacobian.jac .== [0.25,0.05])
@test all(jacobian.invjac .== [4.0,20.0])
@test jacobian.detjac == 0.25*0.05

mesh = UniformMesh([1.0,2.0],[2.0,1.0],[10,3])
quad = TensorProductQuadratureRule(2,4)
jacobian = HDGElasticity.AffineMapJacobian(mesh,quad)
element_size = [2.0/10,1.0/3]
@test all(jacobian.jac .== element_size/2.0)
@test all(jacobian.invjac .== 2.0 ./ element_size)
@test jacobian.detjac == 2.0/120


basis = TensorProductBasis(2,1)
quad = TensorProductQuadratureRule(2,2)
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
