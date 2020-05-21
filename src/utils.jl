function default_integer_type()
    return typeof(1)
end

function default_float_type()
    return typeof(1.0)
end

function get_reference_element_size(::Type{<:ReferenceQuadratureRule})
    return 2.0
end

function number_of_basis_functions(basis::TensorProductBasis{dim,T,NF}) where {dim,T,NF}
    return NF
end

function reference_element(basis::TensorProductBasis{2})
    x0 = [-1.0,-1.0]
    dx = [2.0,2.0]
    return x0,dx
end

function reference_normals(basis::TensorProductBasis{2})
    return [[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]]
end

function jacobian(element_size::SVector{2},quadrature_weights::SVector{N}) where {N}
    return prod(element_size)/sum(quadrature_weights)
end

function jacobian(mesh::UniformMesh{2},quad::TensorProductQuadratureRule{2})
    return jacobian(mesh.element_size,quad.weights)
end

function make_row_matrix(vals::V,matrix::M) where {V<:AbstractVector} where {M<:AbstractMatrix}
    return hcat([v*matrix for v in vals]...)
end

function interpolation_matrix(vals::V,dim::Z) where {V<:AbstractVector,Z<:Integer}
    return make_row_matrix(vals,diagm(ones(dim)))
end

function vec_to_symm_mat_converter(dim::Z) where {Z<:Integer}
    if dim == 2
        E1 = @SMatrix [1.0 0.0
                       0.0 0.0
                       0.0 1.0]
        E2 = @SMatrix [0.0 0.0
                       0.0 1.0
                       1.0 0.0]
        return [E1,E2]
    elseif dim == 3
        E1 = @SMatrix [1.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   1.0   0.0
                       0.0   0.0   1.0
                       0.0   0.0   0.0]
        E2 = @SMatrix [0.0   0.0   0.0
                       0.0   1.0   0.0
                       0.0   0.0   0.0
                       1.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   0.0   1.0]
        E3 = @SMatrix [0.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   0.0   1.0
                       0.0   0.0   0.0
                       1.0   0.0   0.0
                       0.0   1.0   0.0]
        return [E1,E2,E3]
    else
        throw(ArgumentError("Expected dim ∈ {1,2} got dim = $dim"))
    end
end

function symmetric_tensor_dim(dim::Z) where {Z<:Integer}
    if dim == 2
        return 3
    elseif dim == 3
        return 6
    else
        throw(ArgumentError("Expected dim ∈ {2,3}, got dim = $dim"))
    end
end

struct AffineMapJacobian{dim,T}
    jac::SVector{dim,T}
    invjac::SVector{dim,T}
    detjac::T
end

function AffineMapJacobian(jac::Vector{T},invjac::Vector{T},detjac::T) where {T}
    nj = length(jac)
    nij = length(invjac)
    @assert nj == nij
    sjac = SVector{nj}(jac)
    sinvjac = SVector{nij}(invjac)
    return AffineMapJacobian(sjac,sinvjac,detjac)
end

function AffineMapJacobian(element_size::T,reference_element_size::S) where {T<:AbstractVector,S<:Real}
    jac = element_size/reference_element_size
    invjac = inv.(jac)
    detjac = prod(jac)
    return AffineMapJacobian(jac,invjac,detjac)
end

function AffineMapJacobian(element_size::S,quad::TensorProductQuadratureRule{D,T}) where {S<:AbstractVector} where {D,T}
    reference_element_size = get_reference_element_size(T)
    return AffineMapJacobian(element_size,reference_element_size)
end

function AffineMapJacobian(mesh::UniformMesh,quad::TensorProductQuadratureRule)
    return AffineMapJacobian(mesh.element_size,quad)
end
