struct DGMesh{dim,T}
    ncells::Int
    domain::Vector{IntervalBox{dim,T}}
    connectivity::Vector{Vector{Tuple{Int,Int}}}
    cellsign::Vector{Int}
    facemaps::Vector{LineMap{dim,T}}
    facescale::Vector{T}
    cellmap::CellMap{dim,T}
    function DGMesh(domain::Vector{IntervalBox{dim,T}},
        connectivity::Vector{Vector{Tuple{Int,Int}}},
        cellsign::Vector{Int},facemaps::Vector{LineMap{dim,T}},
        facescale::Vector{T},cellmap::CellMap{dim,T}) where {dim,T}

        @assert dim == 2
        nfaces = number_of_faces(dim)

        ncells = length(domain)
        @assert length(connectivity) == ncells
        @assert length(cellsign) == ncells
        @assert length(facemaps) == nfaces

        new{dim,T}(ncells,domain,connectivity,cellsign,facemaps,facescale)

    end
end

function DGMesh(mesh::UniformMesh,coeffs,poly)
    dim = dimension(mesh)
    domain = cell_domain(mesh)
    @assert length(domain) > 0
    cellmap = CellMap(domain[1])
    facescale = face_determinant_jacobian(cellmap)
    connectivity = cell_connectivity(mesh)
    cellsign = cell_signatures(coeffs,poly)
    facemaps = reference_cell_facemaps(dim)
    return DGMesh(domain,connectivity,cellsign,facemaps,facescale,cellmap)
end

function cell_domain(mesh)
    dim = dimension(mesh)
    T = default_float_type()
    domain = Vector{IntervalBox{dim,T}}(undef,mesh.total_number_of_elements)
    for idx in 1:mesh.total_number_of_elements
        xL,xR = CartesianMesh.element(mesh,idx)
        domain[idx] = IntervalBox(xL,xR)
    end
    return domain
end

function cell_connectivity(mesh)
    nfaces = faces_per_cell(mesh)
    ncells = number_of_elements(mesh)
    connectivity = [Vector{Tuple{Int,Int}}(undef,nfaces) for i = 1:ncells]

    for cellid in 1:ncells
        nbrcellids = neighbors(mesh,cellid)
        nbrfaceids = [nc == 0 ? 0 : neighbor_faceid(faceid) for (faceid,nc) in enumerate(nbrcellids)]
        nbrcellandface = collect(zip(nbrcellids,nbrfaceids))
        connectivity[cellid][:] .= nbrcellandface
    end
    return connectivity
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

function number_of_cells(dgmesh::DGMesh)
    return dgmesh.ncells
end
