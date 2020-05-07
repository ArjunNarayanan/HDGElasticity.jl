using Test, StaticArrays, LinearAlgebra
using CartesianMesh, ImplicitDomainQuadrature
using HDGElasticity

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

ALL = HDGElasticity.get_stress_coupling(basis,quad,jac,1)
@test size(ALL) == (4,4)
row1 = [4.0 2.0 2.0 1.0]/9.0
row2 = [2.0 4.0 1.0 2.0]/9.0
row3 = [2.0 1.0 4.0 2.0]/9.0
row4 = [1.0 2.0 2.0 4.0]/9.0
ALLtest = vcat(row1,row2,row3,row4)
@test all(ALL .≈ ALLtest)

ALL = HDGElasticity.get_stress_coupling(basis,quad,jac)
@test size(ALL) == (12,12)

jac = HDGElasticity.AffineMapJacobian([1.0,1.0],quad)
ALL = HDGElasticity.get_stress_coupling(basis,quad,jac,1)
@test size(ALL) == (4,4)
@test all(ALL .≈ 0.25*ALLtest)

jac = HDGElasticity.AffineMapJacobian([1.0,2.0],quad)
ALL = HDGElasticity.get_stress_coupling(basis,quad,jac,1)
@test size(ALL) == (4,4)
@test all(ALL .≈ 0.5*ALLtest)

Dhalf = Array{Float64}(undef,1,1)
Dhalf[1] = 1.0
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
Ek = [Dhalf,Dhalf]
ALUtest = [-2/3 -1/2 -1/2 -1/3
            1/6  0.0  0.0 -1/6
            1/6  0.0  0.0 -1/6
            1/3  1/2  1/2  2/3]
ALU = HDGElasticity.get_stress_displacement_coupling(basis,quad,Dhalf,Ek,jac,1,1)
@test all([isapprox(ALU[i],ALUtest[i],atol=1e-15) for i = 1:length(ALU)])

Dhalf = diagm(ones(3))
ALU = HDGElasticity.get_stress_displacement_coupling(basis,quad,Dhalf,jac)
@test size(ALU) == (12,8)

basis = TensorProductBasis(2,1)
surface_quad = TensorProductQuadratureRule(1,2)
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
AUU = HDGElasticity.get_displacement_coupling(basis,surface_quad,jac,1.0,1)

AUUtest = [0.0  -1/3  1/3  0.0
          -1/3  -4/3  0.0 -1/3
           1/3   0.0  4/3  1/3
           0.0  -1/3  1/3  0.0]
@test all([isapprox(AUU[i],AUUtest[i],atol=1e-15) for i = 1:length(AUU)])

AUU = HDGElasticity.get_displacement_coupling(basis,surface_quad,jac,1.0)
@test size(AUU) == (8,8)

Dhalf = diagm(ones(3))
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
lop = HDGElasticity.LocalOperator(basis,quad,surface_quad,Dhalf,jac,3.0)

@test size(lop.LL) == (12,12)
@test size(lop.LU) == (12,8)
@test size(lop.local_operator) == (20,20)

@test rank(lop.local_operator) == 20
@test !(det(lop.local_operator) ≈ 0.0)
@test norm(lop.local_operator - lop.local_operator') ≈ 0.0
@test issymmetric(lop.local_operator)
