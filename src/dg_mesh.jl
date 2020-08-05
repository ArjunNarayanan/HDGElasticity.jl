struct DGMesh{dim,T}
    domain::Vector{IntervalBox{dim,T}}
    connectivity::Matrix{Tuple{Int,Int}}
    cellsign::Vector{Int}
    function DGMesh(domain::Vector{IntervalBox{dim,T}},
        connectivity::Matrix{Tuple{Int,Int}},
        cellsign::Vector{Int}) where
        {dim,T}

        @assert dim == 2

        ncells = length(domain)
        @assert size(connectivity) == (4,ncells)
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
    connectivity = Matrix{Tuple{Int,Int}}(undef,nfaces,ncells)

    for cellid in 1:ncells
        nbrcellids = neighbors(mesh,cellid)
        nbrfaceids = [nc == 0 ? 0 : neighbor_faceid(faceid) for (faceid,nc) in enumerate(nbrcellids)]
        nbrcellandface = collect(zip(nbrcellids,nbrfaceids))
        connectivity[:,cellid] .= nbrcellandface
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

# function update_active_faces!(isactiveface,faceid,s,cellid)
#     if s == 1
#         isactiveface[faceid,1,cellid] = true
#     elseif s == -1
#         isactiveface[faceid,2,cellid] = true
#     elseif s == 0
#         isactiveface[faceid,1,cellid] = true
#         isactiveface[faceid,2,cellid] = true
#     else
#         throw(ArgumentError("Expected s ∈ {-1,0,+1}, got s = $s"))
#     end
# end

# function active_faces!(isactiveface,isactivecell,coeffs,poly)
#
#     nf,ncells = size(coeffs)
#
#     @assert size(isactivecell) == (2,ncells)
#     @assert size(isactiveface) == (4,2,ncells)
#
#
#     fill!(isactiveface,false)
#
#     dim = dimension(poly)
#     cell = reference_cell(dim)
#     face = reference_cell(dim-1)
#
#     for idx in 1:ncells
#         if isactivecell[1,idx] && !isactivecell[2,idx]
#             isactiveface[:,1,idx] .= true
#         elseif !isactivecell[1,idx] && isactivecell[2,idx]
#             isactiveface[:,2,idx] .= true
#         elseif isactivecell[1,idx] && isactivecell[2,idx]
#
#             update!(poly,coeffs[:,idx])
#             funcs = restrict_on_faces(poly,cell)
#             facesigns = [sign(f,face) for f in funcs]
#
#             for (faceid,fs) in enumerate(facesigns)
#                 update_active_faces!(isactiveface,faceid,fs,idx)
#             end
#
#         end
#     end
# end

# function active_faces(isactivecell,coeffs,poly)
#     nf,ncells = size(coeffs)
#     isactiveface = zeros(Bool,4,2,ncells)
#     active_faces!(isactiveface,isactivecell,coeffs,poly)
#     return isactiveface
# end

# function number_elements!(cell2elid,isactivecell)
#
#     nphase,ncells = size(isactivecell)
#     @assert nphase == 2
#     @assert size(cell2elid) == (2,ncells)
#
#     fill!(cell2elid,0)
#     elid = 1
#     for idx in 1:ncells
#         for phase = 1:2
#             if isactivecell[phase,idx]
#                 cell2elid[phase,idx] = elid
#                 elid += 1
#             end
#         end
#     end
# end

# function number_elements(isactivecell)
#     nphase,ncells = size(isactivecell)
#     cell2elid = zeros(Int,2,ncells)
#     number_elements!(cell2elid,isactivecell)
#     return cell2elid
# end

# function number_face_hybrid_elements!(face2hid,isactiveface,connectivity)
#
#     nface,ncells = size(connectivity)
#     @assert nface == 4
#     @assert size(face2hid) == (4,2,ncells)
#     @assert size(isactiveface) == (4,2,ncells)
#
#     fill!(face2hid,0)
#     hid = 1
#
#     for idx in 1:ncells
#         for phase in 1:2
#             for faceid in 1:4
#                 if isactiveface[faceid,phase,idx] && face2hid[faceid,phase,idx] == 0
#                     face2hid[faceid,phase,idx] = hid
#                     nbr,nbrfaceid = connectivity[faceid,idx]
#                     if nbr != 0
#                         @assert face2hid[nbrfaceid,phase,nbr] == 0
#                         face2hid[nbrfaceid,phase,nbr] = hid
#                     end
#                     hid += 1
#                 end
#             end
#         end
#     end
# end

# function number_face_hybrid_elements(isactiveface,connectivity)
#     face2hid = similar(isactiveface,Int)
#     number_face_hybrid_elements!(face2hid,isactiveface,connectivity)
#     return face2hid
# end

# function number_interface_hybrid_elements!(interface2hid,isactivecell,hid)
#
#     nphase,ncells = size(isactivecell)
#     @assert nphase == 2
#     @assert size(interface2hid) == (2,ncells)
#     @assert hid > 0
#
#     fill!(interface2hid,0)
#
#     for cellid in 1:ncells
#         if isactivecell[1,cellid] && isactivecell[2,cellid]
#             interface2hid[1,cellid] = hid
#             hid += 1
#             interface2hid[2,cellid] = hid
#             hid += 1
#         end
#     end
#
# end

# function number_interface_hybrid_elements(isactivecell,hid)
#     interface2hid = similar(isactivecell,Int)
#     number_interface_hybrid_elements!(interface2hid,isactivecell,hid)
#     return interface2hid
# end
