using Test, StaticArrays, LinearAlgebra
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend
using HDGElasticity

vals = @SVector [0.5,0.5]
direction = [1.0,0.0]
proj = HDGElasticity.normal_basis_projection(vals,direction)
testproj = [0.5,0.0,0.5,0.0]'
@test all(proj .≈ testproj)

surface_basis = TensorProductBasis(1,1)
surface_quad = TensorProductQuadratureRule(1,2)
bcop = HDGElasticity.displacement_component_operator(surface_basis,
    surface_quad,[1.0],1.0,1.0)
bcoptest = 1/3*[2 1
                1 2]
@test all(bcop .≈ bcoptest)

penalty = 10.0
bcop = HDGElasticity.displacement_component_operator(surface_basis,
    surface_quad,[1.0],1.0,penalty)
bcoptest = penalty*1/3*[2 1
                1 2]
@test all(bcop .≈ bcoptest)

jac = 3.0
bcop = HDGElasticity.displacement_component_operator(surface_basis,
    surface_quad,[1.0],jac,1.0)
bcoptest = jac*1/3*[2 1
                    1 2]
@test all(bcop .≈ bcoptest)

direction = [1.0,0.0]
bcop = HDGElasticity.displacement_component_operator(surface_basis,
    surface_quad,direction,1.0,1.0)
bcoptest = 1/3*[2 0 1 0
                0 0 0 0
                1 0 2 0
                0 0 0 0]
@test all(bcop .≈ bcoptest)

rhs = HDGElasticity.displacement_component_rhs(surface_basis,
    surface_quad,1.0,[1.0],1.0,1.0)
rhstest = [1.0,1.0]
@test all(rhs .≈ rhstest)

penalty = 1e2
rhs = HDGElasticity.displacement_component_rhs(surface_basis,
    surface_quad,1.0,[1.0],1.0,penalty)
rhstest = penalty*[1.0,1.0]
@test all(rhs .≈ rhstest)

jac = 0.5
rhs = HDGElasticity.displacement_component_rhs(surface_basis,
    surface_quad,1.0,[1.0],jac,1.0)
rhstest = jac*[1.0,1.0]
@test all(rhs .≈ rhstest)

uD = 1e-2
rhs = HDGElasticity.displacement_component_rhs(surface_basis,surface_quad,
    uD,[1.0],1.0,1.0)
rhstest = uD*[1.0,1.0]
@test all(rhs .≈ rhstest)

uD = 1e-2
penalty = 10.0
rhs = HDGElasticity.displacement_component_rhs(surface_basis,surface_quad,uD,
    [1.0,0.0],1.0,penalty)
rhstest = uD*penalty*[1.0,0.0,1.0,0.0]
@test all(rhs .≈ rhstest)

uD = 1e-2
penalty = 10.0
jac = 0.5
bcop = HDGElasticity.DisplacementComponentBC(surface_basis,surface_quad,
    uD,[0.0,1.0],jac,penalty=penalty)
optest = penalty*jac*1/3*[0 0 0 0
                          0 2 0 1
                          0 0 0 0
                          0 1 0 2]
@test all(bcop.op .≈ optest)

rhstest = uD*penalty*jac*[0.,1,0,1]
@test all(bcop.rhs .≈ rhstest)
