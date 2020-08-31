function evaluate_levelset(levelset,coords::M) where {M<:AbstractMatrix}
    return [levelset(view(coords,:,i)) for i in 1:size(coords)[2]]
end

function evaluate_multiple_levelsets(levelset,coeffs::M,refpoints) where {M<:AbstractMatrix}
    ft = default_float_type()
    numcells = size(coeffs)[2]
    vals = Vector{Vector{ft}}(undef,numcells)
    for cellid in 1:numcells
        update!(levelset,coeffs[:,cellid])
        vals[cellid] = evaluate_levelset(levelset,refpoints)
    end
    return vals
end

function transfer_levelset(levelset,coeffs,cellmaps,basis,quad,mass_matrix,nf)
    @assert size(coeffs)[2] == length(cellmaps)
    rhs = zeros(nf)
    levelset_vals = evaluate_multiple_levelsets(levelset,coeffs,quad.points)
    for (cellmap,val) in zip(cellmaps,levelset_vals)
        linear_form!(rhs,val',basis,cellmap,quad)
    end
    return mass_matrix\rhs
end

function transfer_levelset(levelset,coeffs,cellmaps,basis,quad,mass_matrix)
    nf = number_of_basis_functions(basis)
    return transfer_levelset(levelset,coeffs,cellmaps,basis,quad,mass_matrix,nf)
end
