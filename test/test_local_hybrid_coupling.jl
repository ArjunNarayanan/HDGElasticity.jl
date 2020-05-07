using Test, StaticArrays, LinearAlgebra
using CartesianMesh, ImplicitDomainQuadrature
using HDGElasticity
import ImplicitDomainQuadrature: extend

basis = TensorProductBasis(2,1)
surface_basis = TensorProductBasis(1,1)
quad = TensorProductQuadratureRule(2,2)
surface_quad = TensorProductQuadratureRule(1,2)
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)

LUh = zeros(4,2)
M = Matrix{Float64}(undef,1,1)
M[1] = 1.0
HDGElasticity.update_stress_hybrid_coupling!(LUh,x->basis(extend(x,2,-1.0)),
    surface_basis,M,surface_quad,jac.jac[1],1)

LUhtest = [2/3  1/3
           0.0  0.0
           1/3  2/3
           0.0  0.0]
@test all([isapprox(LUh[i],LUhtest[i]) for i in 1:length(LUh)])

LUh = zeros(4,2)
testjac = 3.0
HDGElasticity.update_stress_hybrid_coupling!(LUh,x->basis(extend(x,1,1.0)),
    surface_basis,M,surface_quad,testjac,1)
LUhtest = testjac*[0.0  0.0
                   0.0  0.0
                   2/3  1/3
                   1/3  2/3]
@test all([isapprox(LUh[i],LUhtest[i]) for i in 1:length(LUh)])

LUh = zeros(4,2)
HDGElasticity.update_stress_hybrid_coupling!(LUh,x->basis(extend(x,2,1.0)),
    surface_basis,M,surface_quad,-1.0,1)
LUhtest = [0.0   0.0
          -2/3  -1/3
           0.0   0.0
          -1/3  -2/3]
@test all([isapprox(LUh[i],LUhtest[i]) for i = 1:length(LUh)])
