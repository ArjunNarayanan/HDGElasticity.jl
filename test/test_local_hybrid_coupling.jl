using Test
using LinearAlgebra
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend
using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

basis = TensorProductBasis(2,1)
sbasis = TensorProductBasis(1,1)
quad = tensor_product_quadrature(2,2)
squad = tensor_product_quadrature(1,2)

LH = zeros(4,2)
M = [1.0]
HDGElasticity.update_LHop!(LH,x->basis(extend(x,2,-1.0)),sbasis,squad,M,1.0,1)
LHtest =  [2/3  1/3
           0.0  0.0
           1/3  2/3
           0.0  0.0]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
testjac = 3.0
HDGElasticity.update_LHop!(LH,x->basis(extend(x,1,1.0)),sbasis,squad,M,testjac,1)
LHtest =  testjac*[0.0  0.0
                   0.0  0.0
                   2/3  1/3
                   1/3  2/3]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
HDGElasticity.update_LHop!(LH,x->basis(extend(x,2,1.0)),sbasis,squad,M,+1.0,1)
LHtest = [0.0   0.0
          +2/3  +1/3
           0.0   0.0
          +1/3  +2/3]
@test allapprox(LH,LHtest)

LH = zeros(4,2)
HDGElasticity.update_LHop!(LH,x->basis(extend(x,1,-1.0)),sbasis,squad,M,+1.0,1)
LHtest =  [+2/3  +1/3
           +1/3  +2/3
            0.0   0.0
            0.0   0.0]
@test allapprox(LH,LHtest)

Ek = HDGElasticity.vec_to_symm_mat_converter(2)
Dhalf = diagm(ones(3))
LH1 = HDGElasticity.LHop(x->basis(extend(x,2,-1.0)),
    sbasis,squad,[0.0,-1.0],Dhalf,Ek,0.5,2,3,4,2)
LH2 = HDGElasticity.LHop(x->basis(extend(x,1,+1.0)),
    sbasis,squad,[1.0,0.0],Dhalf,Ek,1.0,2,3,4,2)
LH3 = HDGElasticity.LHop(x->basis(extend(x,2,+1.0)),
    sbasis,squad,[0.0,1.0],Dhalf,Ek,+0.5,2,3,4,2)
LH4 = HDGElasticity.LHop(x->basis(extend(x,1,-1.0)),
    sbasis,squad,[-1.0,0.0],Dhalf,Ek,+1.0,2,3,4,2)

cellmap = HDGElasticity.AffineMap([0.,0.],[1.,2.])
LH = HDGElasticity.LHop(basis,sbasis,squad,Dhalf,cellmap)

@test allapprox(LH[1],LH1)
@test allapprox(LH[2],LH2)
@test allapprox(LH[3],LH3)
@test allapprox(LH[4],LH4)

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
