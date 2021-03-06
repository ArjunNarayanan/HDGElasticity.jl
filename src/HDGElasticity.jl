module HDGElasticity

using LinearAlgebra
using SparseArrays
using StaticArrays
using IntervalArithmetic
using Roots
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend

IDQ = ImplicitDomainQuadrature

include("affine_map.jl")
include("utils.jl")
include("variational_form_utils.jl")
include("dg_mesh.jl")
include("fit_interface_hybrid_element.jl")
include("function_space.jl")
include("isotropic_elasticity.jl")
include("local_operator.jl")
include("local_hybrid_coupling.jl")
include("hybrid_coupling.jl")
include("hybrid_local_coupling.jl")
include("local_solver.jl")
include("assembly.jl")


end # module
