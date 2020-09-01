struct BackgroundMesh{dim}
    ncells::Int
    domain::Vector{IntervalBox{dim}}
    connectivity::Matrix{Tuple{Int,Int}}
    reference_facemaps::Vector{LineMap{dim}}
    function BackgroundMesh(domain::Vector{IntervalBox{dim,T}},
        connectivity::Matrix{Tuple{Int,Int}},
        reference_facemaps::Vector{LineMap{dim,T}}) where {dim,T}

        @assert dim == 2

        nfaces = number_of_faces(dim)
        ncells = length(domain)
        @assert size(connectivity) == (nfaces,ncells)
        @assert length(reference_facemaps) == nfaces

        new{dim}(ncells,domain,connectivity,reference_facemaps)
    end
end

function BackgroundMesh(mesh)
    dim = dimension(mesh)
    domains = cell_domain(mesh)
    connectivity = cell_connectivity(mesh)
    reference_facemaps = reference_cell_facemaps(dim)
    return BackgroundMesh(domains,connectivity,reference_facemaps)
end

function cell_domain(mesh)
    dim = dimension(mesh)
    T = default_float_type()
    ncells = number_of_elements(mesh)
    domain = Vector{IntervalBox{dim,T}}(undef,ncells)
    for idx in 1:ncells
        xL,xR = element(mesh,idx)
        domain[idx] = IntervalBox(xL,xR)
    end
    return domain
end

function cell_connectivity(mesh)
    nfaces = faces_per_cell(mesh)
    ncells = number_of_elements(mesh)
    connectivity = Matrix{Tuple{Int,Int}}(undef,nfaces,ncells)

    for cellid in 1:ncells
        nbrcellids = neighbors(mesh,cellid)
        nbrfaceids = [nc == 0 ? 0 : neighbor_faceid(faceid) for (faceid,nc) in enumerate(nbrcellids)]
        nbrcellandface = collect(zip(nbrcellids,nbrfaceids))
        connectivity[:,cellid] .= nbrcellandface
    end
    return connectivity
end
