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

function update_mass_matrix!(matrix,basis,quad,ndofs,scale)
    for (p,w) in quad
        vals = basis(p)
        N = interpolation_matrix(vals,ndofs)
        matrix .+= scale*N'*N*w
    end
end

function update_mass_matrix!(matrix,basis,quad,map::InterpolatingPolynomial,
    ndofs,scale)
    
    @assert length(scale) == length(quad)
    for (idx,(p,w)) in enumerate(quad)
        vals = basis(map(p))
        detjac = determinant_jacobian(map,p)
        N = interpolation_matrix(vals,ndofs)
        matrix .+= N'*N*detjac*scale[idx]*w
    end
end

function update_mass_matrix_on_active_faces!(matrix,basis,facequads,
    isactiveface,ndofs,cellmap)

    dim = dimension(basis)
    cell = reference_cell(dim)

    funcs = restrict_on_faces(basis,cell)
    jac = jacobian(cellmap,cell)

    for (faceid,func) in enumerate(funcs)
        if isactiveface[faceid]
            update_mass_matrix!(matrix,func,facequads[faceid],ndofs,jac[faceid])
        end
    end

end

function mass_matrix(basis,quad,ndofs,scale)
    nf = number_of_basis_functions(basis)
    nfndofs = nf*ndofs
    matrix = zeros(nfndofs,nfndofs)
    update_mass_matrix!(matrix,basis,quad,ndofs,scale)
    return matrix
end

# function mass_matrix_on_boundary(basis,quad,ndofs,cellmap)
#
#     dim = dimension(basis)
#     NF = number_of_basis_functions(basis)
#     cell = reference_cell(dim)
#
#     totaldofs = ndofs*NF
#     matrix = zeros(totaldofs,totaldofs)
#
#     funcs = restrict_on_faces(basis,cell)
#     jac = jacobian(cellmap,cell)
#
#     for (faceid,func) in enumerate(funcs)
#         update_mass_matrix!(matrix,func,quad,ndofs,jac[faceid])
#     end
#
#     return matrix
# end

function mass_matrix_on_boundary(basis,facequads,isactiveface,ndofs,cellmap)

    dim = dimension(basis)
    NF = number_of_basis_functions(basis)
    cell = reference_cell(dim)

    totaldofs = ndofs*NF
    matrix = zeros(totaldofs,totaldofs)

    funcs = restrict_on_faces(basis,cell)
    jac = jacobian(cellmap,cell)

    nfaces = length(funcs)
    @assert length(facequads) == nfaces
    @assert length(isactiveface) == nfaces

    for faceid in 1:nfaces
        if isactiveface[faceid]
            update_mass_matrix!(matrix,funcs[faceid],
                facequads[faceid],ndofs,jac[faceid])
        end
    end

    return matrix

end

function mass_matrix_on_boundary(basis,facequads,isactiveface,iquad,
    normals,imap,ndofs,cellmap)

    dim = dimension(basis)
    NF = number_of_basis_functions(basis)
    cell = reference_cell(dim)

    totaldofs = ndofs*NF
    matrix = zeros(totaldofs,totaldofs)

    update_mass_matrix_on_active_faces!(matrix,basis,facequads,
        isactiveface,ndofs,cellmap)
    scale = scale_area(cellmap,normals)
    update_mass_matrix!(matrix,basis,iquad,imap,ndofs,scale)

    return matrix

end

function linear_form(rhsvals::M,basis,quad) where {M<:AbstractMatrix}
    ndofs,nq = size(rhsvals)
    @assert number_of_quadrature_points(quad) == nq

    nf = number_of_basis_functions(basis)
    rhs = zeros(nf*ndofs)

    for (idx,(p,w)) in enumerate(quad)
        vals = basis(p)
        N = interpolation_matrix(vals,ndofs)
        rhs += N'*rhsvals[:,idx]*w
    end
    return rhs
end
