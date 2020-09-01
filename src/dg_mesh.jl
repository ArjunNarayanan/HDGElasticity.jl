struct DGMesh{dim,T}
    domain::Vector{IntervalBox{dim,T}}
    connectivity::Vector{Vector{Tuple{Int,Int}}}
    cellsign::Vector{Int}
    function DGMesh(domain::Vector{IntervalBox{dim,T}},
        connectivity::Vector{Vector{Tuple{Int,Int}}},
        cellsign::Vector{Int}) where
        {dim,T}

        @assert dim == 2

        ncells = length(domain)
        @assert length(connectivity) == ncells
        @assert length(cellsign) == ncells

        new{dim,T}(domain,connectivity,cellsign)

    end
end

function DGMesh(mesh::UniformMesh,coeffs,poly)
    domain = cell_domain(mesh)
    connectivity = cell_connectivity(mesh)
    cellsign = cell_signatures(coeffs,poly)
    return DGMesh(domain,connectivity,cellsign)
end

function cell_signatures!(cellsign,coeffs,poly)

    nf,ncells = size(coeffs)

    @assert length(cellsign) == ncells

    dim = dimension(poly)
    box = reference_cell(dim)
    fill!(cellsign,0)

    for idx in 1:ncells
        update!(poly,coeffs[:,idx])
        cellsign[idx] = sign(poly,box)
    end
end

function cell_signatures(coeffs,poly)
    nf,ncells = size(coeffs)
    cellsign = zeros(Int,ncells)
    cell_signatures!(cellsign,coeffs,poly)
    return cellsign
end

function dimension(dgmesh::DGMesh{dim}) where dim
    return dim
end
