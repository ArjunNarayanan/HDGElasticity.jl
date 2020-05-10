struct DisplacementComponentBC{T}
    op::Matrix{T}
    rhs::Vector{T}
end

function DisplacementComponentBC(surface_basis::TensorProductBasis{1},
    surface_quad::TensorProductQuadratureRule{1},displacement::S,
    direction::Vector{S},jac::S;penalty::S=1e2) where {S<:Real}

    # @assert length(direction) == 2
    @assert norm(direction) ≈ 1.0
    
    op = displacement_component_operator(surface_basis,surface_quad,direction,jac,penalty)
    rhs = displacement_component_rhs(surface_basis,surface_quad,displacement,direction,jac,penalty)

    return DisplacementComponentBC(op,rhs)
end

function normal_basis_projection(vals::SVector{N,T},direction::Vector{T}) where {N,T}
    return hcat([v*direction' for v in vals]...)
end

function displacement_component_operator(surface_basis::TensorProductBasis{1,T,NF},
    surface_quad::TensorProductQuadratureRule{1},direction::Vector{S},
    jac::S,penalty::S) where {T,NF,S<:Real}

    dim = length(direction)

    bc_op = zeros(dim*NF,dim*NF)

    for (p,w) in surface_quad
        svals = surface_basis(p)

        Nn = normal_basis_projection(svals,direction)

        bc_op += penalty*Nn'*Nn*jac*w
    end
    return bc_op
end

function displacement_component_rhs(surface_basis::TensorProductBasis{1,T,NF},
    surface_quad::TensorProductQuadratureRule{1},displacement::S,
    direction::Vector{S},jac::S,penalty::S) where {T,NF,S<:Real}

    dim = length(direction)

    bc_rhs = zeros(dim*NF)

    for (p,w) in surface_quad
        svals = surface_basis(p)

        rhs = displacement*vcat([v*direction for v in svals]...)
        bc_rhs += penalty*rhs*jac*w
    end
    return bc_rhs
end
