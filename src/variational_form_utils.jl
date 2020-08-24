function make_row_matrix(vals,matrix)
    return hcat([v*matrix for v in vals]...)
end

function interpolation_matrix(vals,ndofs)
    return make_row_matrix(vals,diagm(ones(ndofs)))
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

function symmetric_tensor_dimension(dim)
    if dim == 2
        return 3
    elseif dim == 3
        return 6
    else
        throw(ArgumentError("Expected dim ∈ {2,3}, got dim = $dim"))
    end
end

function update_mass_matrix!(matrix,basis,quad,scale,ndofs)
    for (p,w) in quad
        vals = basis(p)
        N = interpolation_matrix(vals,ndofs)
        matrix .+= scale*N'*N*w
    end
end

function update_mass_matrix!(matrix,basis,quad,linemap::LineMap,scale,ndofs)
    detjac = determinant_jacobian(linemap)
    for (p,w) in quad
        vals = basis(linemap(p...))
        N = interpolation_matrix(vals,ndofs)
        matrix .+= N'*N*detjac*scale*w
    end
end

function update_mass_matrix!(matrix,basis,quad,map::InterpolatingPolynomial,
    scale,ndofs)

    @assert length(scale) == length(quad)
    for (idx,(p,w)) in enumerate(quad)
        vals = basis(map(p))
        detjac = determinant_jacobian(map,p)
        N = interpolation_matrix(vals,ndofs)
        matrix .+= N'*N*detjac*scale[idx]*w
    end
end

function update_mass_matrix_on_faces!(matrix,basis,facequads,facemaps,
    scale,ndofs)

    @assert length(facequads) == length(facemaps)
    @assert length(facequads) == length(scale)
    for (quad,map,s) in zip(facequads,facemaps,scale)
        update_mass_matrix!(matrix,basis,quad,
            map,s,ndofs)
    end
end

function mass_matrix(basis,quad,scale,ndofs)
    nf = number_of_basis_functions(basis)
    nfndofs = nf*ndofs
    matrix = zeros(nfndofs,nfndofs)
    update_mass_matrix!(matrix,basis,quad,scale,ndofs)
    return matrix
end

function mass_matrix_on_boundary(basis,facequads,facemaps,scale,ndofs,NF)

    totaldofs = ndofs*NF
    matrix = zeros(totaldofs,totaldofs)
    update_mass_matrix_on_faces!(matrix,basis,facequads,facemaps,scale,ndofs)

    return matrix
end


function mass_matrix_on_boundary(basis,facequads,facemaps,scale,ndofs)

    NF = number_of_basis_functions(basis)
    return mass_matrix_on_boundary(basis,facequads,facemaps,scale,ndofs,NF)
end

function mass_matrix_on_boundary(basis,facequads,facemaps,facescale,
    iquad,imap,iscale,ndofs,nf)

    totaldofs = ndofs*nf
    matrix = zeros(totaldofs,totaldofs)

    update_mass_matrix_on_faces!(matrix,basis,facequads,
        facemaps,facescale,ndofs)
    update_mass_matrix!(matrix,basis,iquad,imap,iscale,ndofs)

    return matrix

end

# 
# function mass_matrix_on_boundary(basis,facequads,facemaps,facescale,
#     iquad,imap,iscale,ndofs)
#
#     nf = number_of_basis_functions(basis)
#     matrix = mass_matrix_on_boundary(basis,facequads,facemaps,facescale,
#         iquad,imap,iscale,ndofs,nf)
#
#     return matrix
#
# end

function linear_form(rhsvals::M,basis,quad) where {M<:AbstractMatrix}
    ndofs,nq = size(rhsvals)
    @assert length(quad) == nq

    nf = number_of_basis_functions(basis)
    rhs = zeros(nf*ndofs)

    for (idx,(p,w)) in enumerate(quad)
        vals = basis(p)
        N = interpolation_matrix(vals,ndofs)
        rhs += N'*rhsvals[:,idx]*w
    end
    return rhs
end

function linear_form(rhsvals::V,basis,quad) where {V<:AbstractVector}

    nq = length(quad)
    extrhs = repeat(rhsvals,inner=(1,nq))
    return linear_form(extrhs,basis,quad)
end
