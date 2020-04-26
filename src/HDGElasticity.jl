module HDGElasticity

using LinearAlgebra
using StaticArrays
using CartesianMesh
using ImplicitDomainQuadrature

include("local_operator.jl")
include("local_hybrid_coupling.jl")

end # module
