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
