function default_integer_type()
    return typeof(1)
end

function default_float_type()
    return typeof(1.0)
end

function reference_element_length()
    return 2.0
end

function number_of_basis_functions(basis::TensorProductBasis{dim,T,NF}) where
    {dim,T,NF}

    return NF
end

function reference_cell(basis::TensorProductBasis{2})
    xL = [-1.0,-1.0]
    xR = [+1.0,+1.0]
    return xL,xR
end

function reference_normals()
    return [[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]]
end

function reference_cell_volume(dim)
    return reference_element_length()^dim
end

function affine_map_jacobian(element_size)
    dim = length(element_size)
    reference_vol = reference_cell_volume(dim)
    return prod(element_size)/reference_vol
end

function make_row_matrix(vals::V,
        matrix::M) where {V<:AbstractVector} where {M<:AbstractMatrix}

    return hcat([v*matrix for v in vals]...)
end

function interpolation_matrix(vals::V,
        dim::Z) where {V<:AbstractVector,Z<:Integer}

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

struct AffineMap{dim,T}
    xL::SVector{dim,T}
    xR::SVector{dim,T}
    function AffineMap(xL::SVector{dim,T},xR::SVector{dim,T}) where {dim,T}
        @assert 1 <= dim <= 3
        @assert all(xL .< xR)
        new{dim,T}(xL,xR)
    end
end

function AffineMap(xL,xR)
    dim = length(xL)
    @assert length(xR) == dim
    sxL = SVector{dim}(xL)
    sxR = SVector{dim}(xR)
    return AffineMap(sxL,sxR)
end

function (M::AffineMap{dim})(xi) where {dim}
    @assert length(xi) == dim
    return M.xL .+ 0.5*(1.0 .+ xi) .* (M.xR - M.xL)
end
