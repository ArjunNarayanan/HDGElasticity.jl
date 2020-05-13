using Test
using ImplicitDomainQuadrature
using HDGElasticity

basis = TensorProductBasis(2,4)
@test HDGElasticity.number_of_basis_functions(basis) == 25

@test_throws ArgumentError HDGElasticity.symmetric_tensor_dim(1)
@test_throws ArgumentError HDGElasticity.symmetric_tensor_dim(4)
