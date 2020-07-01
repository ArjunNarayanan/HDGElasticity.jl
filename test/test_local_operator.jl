using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using HDGElasticity

basis = TensorProductBasis(2,1)
quad = TensorProductQuadratureRule(2,2)
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
ALL = HDGElasticity.get_stress_coupling(basis,quad,jac,1)
@test size(ALL) == (4,4)
ALLtest = 1.0/9.0*[4.   2   2   1
                   2    4   1   2
                   2    1   4   2
                   1    2   2   4]
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

ALL = HDGElasticity.get_stress_coupling(basis,quad,jac)
@test size(ALL) == (12,12)

basis2 = TensorProductBasis(2,2)
quad2 = TensorProductQuadratureRule(2,4)
ALL = HDGElasticity.get_stress_coupling(basis2,quad2,jac)
@test size(ALL) == (27,27)

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

ALU = HDGElasticity.get_stress_displacement_coupling(basis2,quad2,Dhalf,jac)
@test size(ALU) == (27,18)

basis = TensorProductBasis(2,1)
surface_quad = TensorProductQuadratureRule(1,2)
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
AUU = HDGElasticity.get_displacement_coupling(basis,surface_quad,jac,1.0,1)

AUUtest = [+4/3  +1/3  +1/3  +0.0
           +1/3  +4/3  +0.0  +1/3
           +1/3   0.0  +4/3  +1/3
           +0.0  +1/3  +1/3  +4/3]
@test all([isapprox(AUU[i],AUUtest[i],atol=1e-15) for i = 1:length(AUU)])

AUU = HDGElasticity.get_displacement_coupling(basis,surface_quad,jac,1.0)
@test size(AUU) == (8,8)

Dhalf = diagm(ones(3))
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
lop = HDGElasticity.LocalOperator(basis,quad,surface_quad,Dhalf,jac,3.0)

@test size(lop.LL) == (12,12)
@test size(lop.LU) == (12,8)
@test size(lop.UU) == (8,8)
@test size(lop.local_operator) == (20,20)

@test rank(lop.local_operator) == 20
@test !(det(lop.local_operator) ≈ 0.0)
@test norm(lop.local_operator - lop.local_operator') ≈ 0.0
@test issymmetric(lop.local_operator)

x0 = [0.0,0.0]
widths = [2.0,1.0]
nelements = [1,1]
mesh = UniformMesh(x0,widths,nelements)
basis = TensorProductBasis(2,2)
surface_basis = TensorProductBasis(1,2)
quad = TensorProductQuadratureRule(2,2)
surface_quad = TensorProductQuadratureRule(1,2)
jac = HDGElasticity.AffineMapJacobian(mesh,quad)
@test_throws ArgumentError HDGElasticity.LocalOperator(surface_basis,
    surface_quad,surface_quad,Dhalf,jac,3.0)
