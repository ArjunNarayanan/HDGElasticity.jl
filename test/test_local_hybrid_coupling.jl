using Test, StaticArrays, LinearAlgebra
using CartesianMesh, ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend
using HDGElasticity

a = Vector{Matrix{Float64}}(undef,0)
@test_throws AssertionError HDGElasticity.check_all_matrix_sizes(a)

a = [rand(2,3),rand(2,2)]
@test_throws AssertionError HDGElasticity.check_all_matrix_sizes(a)

basis = TensorProductBasis(2,1)
surface_basis = TensorProductBasis(1,1)
quad = TensorProductQuadratureRule(2,2)
surface_quad = TensorProductQuadratureRule(1,2)
jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)

LUh = zeros(4,2)
M = Matrix{Float64}(undef,1,1)
M[1] = 1.0
HDGElasticity.update_stress_hybrid_coupling!(LUh,x->basis(extend(x,2,-1.0)),
    surface_basis,surface_quad,M,jac.jac[1],1)

LUhtest = [2/3  1/3
           0.0  0.0
           1/3  2/3
           0.0  0.0]
@test all([isapprox(LUh[i],LUhtest[i]) for i in 1:length(LUh)])

LUh = zeros(4,2)
testjac = 3.0
HDGElasticity.update_stress_hybrid_coupling!(LUh,x->basis(extend(x,1,1.0)),
    surface_basis,surface_quad,M,testjac,1)
LUhtest = testjac*[0.0  0.0
                   0.0  0.0
                   2/3  1/3
                   1/3  2/3]
@test all([isapprox(LUh[i],LUhtest[i]) for i in 1:length(LUh)])

LUh = zeros(4,2)
HDGElasticity.update_stress_hybrid_coupling!(LUh,x->basis(extend(x,2,1.0)),
    surface_basis,surface_quad,M,+1.0,1)
LUhtest = [0.0   0.0
          +2/3  +1/3
           0.0   0.0
          +1/3  +2/3]
@test all([isapprox(LUh[i],LUhtest[i]) for i = 1:length(LUh)])

LUh = zeros(4,2)
HDGElasticity.update_stress_hybrid_coupling!(LUh,x->basis(extend(x,1,-1.0)),
    surface_basis,surface_quad,M,+1.0,1)
LUhtest = [+2/3  +1/3
           +1/3  +2/3
            0.0   0.0
            0.0   0.0]
@test all([isapprox(LUh[i],LUhtest[i]) for i = 1:length(LUh)])

Dhalf = diagm(ones(3))
LUh1 = HDGElasticity.get_stress_hybrid_coupling(x->basis(extend(x,2,-1.0)),
    surface_basis,surface_quad,[0.0,-1.0],Dhalf,0.5,2,3,4,2)
LUh2 = HDGElasticity.get_stress_hybrid_coupling(x->basis(extend(x,1,+1.0)),
    surface_basis,surface_quad,[1.0,0.0],Dhalf,1.0,2,3,4,2)
LUh3 = HDGElasticity.get_stress_hybrid_coupling(x->basis(extend(x,2,+1.0)),
    surface_basis,surface_quad,[0.0,1.0],Dhalf,+0.5,2,3,4,2)
LUh4 = HDGElasticity.get_stress_hybrid_coupling(x->basis(extend(x,1,-1.0)),
    surface_basis,surface_quad,[-1.0,0.0],Dhalf,+1.0,2,3,4,2)

jac = HDGElasticity.AffineMapJacobian([1.0,2.0],quad)
normals = [[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]]
LUh = HDGElasticity.get_stress_hybrid_coupling(basis,surface_basis,surface_quad,
    Dhalf,jac,[-1.0,-1.0],[2.0,2.0],normals)

@test all(LUh[1] .≈ LUh1)
@test all(LUh[2] .≈ LUh2)
@test all(LUh[3] .≈ LUh3)
@test all(LUh[4] .≈ LUh4)

LUh = HDGElasticity.get_stress_hybrid_coupling(basis,surface_basis,surface_quad,
    Dhalf,jac)
@test all(LUh[1] .≈ LUh1)
@test all(LUh[2] .≈ LUh2)
@test all(LUh[3] .≈ LUh3)
@test all(LUh[4] .≈ LUh4)

UUh = HDGElasticity.get_displacement_hybrid_coupling(x->basis(extend(x,2,-1.0)),
    surface_basis,surface_quad,1.0,1.0,1,4,2)
UUhtest = [2/3  1/3
           0.0  0.0
           1/3  2/3
           0.0  0.0]
@test all([isapprox(UUh[i],UUhtest[i]) for i = 1:length(UUh)])

testjac = 2.0
UUh = HDGElasticity.get_displacement_hybrid_coupling(x->basis(extend(x,1,+1.0)),
    surface_basis,surface_quad,1.0,testjac,1,4,2)
UUhtest = testjac*[0.0  0.0
                   0.0  0.0
                   2/3  1/3
                   1/3  2/3]
@test all([isapprox(UUh[i],UUhtest[i]) for i = 1:length(UUh)])

UUh = HDGElasticity.get_displacement_hybrid_coupling(x->basis(extend(x,2,+1.0)),
    surface_basis,surface_quad,1.0,+1.0,1,4,2)
UUhtest = [0.0   0.0
          +2/3  +1/3
           0.0   0.0
          +1/3  +2/3]
@test all(UUh .≈ UUhtest)

UUh = HDGElasticity.get_displacement_hybrid_coupling(x->basis(extend(x,1,-1.0)),
    surface_basis,surface_quad,1.0,+1.0,1,4,2)
UUhtest = [+2/3  +1/3
           +1/3  +2/3
            0.0   0.0
            0.0   0.0]
@test all(UUh .≈ UUhtest)

jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
UUh = HDGElasticity.get_displacement_hybrid_coupling(basis,surface_basis,
    surface_quad,1.0,jac,[-1.0,-1.0],[2.0,2.0])
s = size.(UUh)
@test all([i == (8,4) for i in s])

UUh = HDGElasticity.get_displacement_hybrid_coupling(basis,surface_basis,
    surface_quad,3.0,jac)
s = size.(UUh)
@test all([i == (8,4) for i in s])

HH = HDGElasticity.get_hybrid_coupling(surface_basis,surface_quad,1.0,1.0,1,2)
HHtest = 1/3*[2 1
              1 2]
@test all(HH .≈ HHtest)

HH = HDGElasticity.get_hybrid_coupling(surface_basis,surface_quad,1.0,-1.0,1,2)
HHtest = -1/3*[2 1
              1 2]
@test all(HH .≈ HHtest)

teststab = 3.0
HH = HDGElasticity.get_hybrid_coupling(surface_basis,surface_quad,teststab,-1.0,1,2)
HHtest = -teststab*1/3*[2 1
                        1 2]
@test all(HH .≈ HHtest)

jac = HDGElasticity.AffineMapJacobian([2.0,2.0],quad)
HH = HDGElasticity.get_hybrid_coupling(surface_basis,surface_quad,3.0,jac)

s = size.(HH)
@test all([i == (4,4) for i in s])

local_hybrid = HDGElasticity.LocalHybridCoupling(basis,surface_basis,surface_quad,
    Dhalf,jac,3.0)
