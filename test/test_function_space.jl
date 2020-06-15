using Test
using PolynomialBasis
using CartesianMesh
using Revise
using HDGElasticity

function allequal(u,v)
    return all(u .== v)
end

tpb = TensorProductBasis(2,2)
basis = HDGElasticity.bases(tpb,5)
@test typeof(basis) == Vector{TensorProductBasis{2,LagrangePolynomialBasis{3},9}}

basis = HDGElasticity.bases(2,3,5)
@test typeof(basis) == Vector{TensorProductBasis{2,LagrangePolynomialBasis{4},16}}

orders = [2,3,1,2]
basis = HDGElasticity.bases(2,orders)
testnf = (orders .+ 1).^2
nf = HDGElasticity.number_of_basis_functions.(basis)
@test allequal(testnf,nf)
