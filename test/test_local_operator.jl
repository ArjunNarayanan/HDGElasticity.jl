using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allapprox(v1,v2,tol)
    @assert length(v1) == length(v2)
    return all([isapprox(v1[i],v2[i],atol=tol) for i = 1:length(v1)])
end

basis = TensorProductBasis(2,1)
quad = tensor_product_quadrature(2,2)
ALL = HDGElasticity.LLop(basis,quad)
rows = 1.0/9.0*[4.   2   2   1
                2    4   1   2
                2    1   4   2
                1    2   2   4]
ALLtest = vcat([HDGElasticity.interpolation_matrix(rows[i,:],3) for i = 1:4]...)
@test allapprox(ALL,ALLtest)

map = HDGElasticity.AffineMap([0.0,0.0],[1.0,1.0])
ALL = HDGElasticity.LLop(basis,quad,map)
@test allapprox(ALL,0.25*ALLtest)

Dhalf = reshape([1.0],1,1)
map = HDGElasticity.AffineMap([-1.,-1.],[1.,1.])
Ek = [Dhalf,Dhalf]
ALUtest = [-2/3 -1/2 -1/2 -1/3
            1/6  0.0  0.0 -1/6
            1/6  0.0  0.0 -1/6
            1/3  1/2  1/2  2/3]
ALU = HDGElasticity.LUop(basis,quad,Dhalf,map,Ek)
@test allapprox(ALU,ALUtest,1e-15)

Dhalf = diagm(ones(3))
ALU = HDGElasticity.LUop(basis,quad,Dhalf,map)
@test size(ALU) == (12,8)

basis2 = TensorProductBasis(2,2)
quad2 = tensor_product_quadrature(2,3)
ALU = HDGElasticity.LUop(basis2,quad2,Dhalf,map)
@test size(ALU) == (27,18)

basis = TensorProductBasis(2,1)
surface_quad = tensor_product_quadrature(1,2)
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
