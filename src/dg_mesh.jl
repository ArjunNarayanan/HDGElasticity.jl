struct DGMesh{dim,T}
    domain::Vector{IntervalBox{dim,T}}
    connectivity::Matrix{Tuple{Int,Int}}
    isactivecell::Matrix{Bool}
    isactiveface::Array{Bool,3}
    cell2elid::Matrix{Int}
    face2hid::Array{Int,3}
    interface2hid::Matrix{Int}
    elids::UnitRange{Int}
    facehids::UnitRange{Int}
    interfacehids::UnitRange{Int}
    function DGMesh(domain::Vector{IntervalBox{dim,T}},
        connectivity::Matrix{Tuple{Int,Int}},
        isactivecell::Matrix{Bool},isactiveface::Array{Bool,3},
        cell2elid::Matrix{Int},face2hid::Array{Int,3},
        interface2hid::Matrix{Int}) where
        {dim,T}

        @assert dim == 2

        ncells = length(domain)
        @assert size(connectivity) == (4,ncells)
        @assert size(isactivecell) == (2,ncells)
        @assert size(isactiveface) == (4,2,ncells)
        @assert size(cell2elid) == (2,ncells)
        @assert size(face2hid) == (4,2,ncells)
        @assert size(interface2hid) == (2,ncells)

        elidstop = maximum(cell2elid)
        elids = 1:elidstop
        facehidstop = maximum(face2hid)
        facehids = 1:facehidstop
        interfacehidstop = maximum(interface2hid)
        interfacehids = (facehidstop+1):interfacehidstop

        new{dim,T}(domain,connectivity,isactivecell,isactiveface,
            cell2elid,face2hid,interface2hid,elids,facehids,interfacehids)

    end
end

function DGMesh(mesh::UniformMesh,coeffs,poly)
    domain = cell_domain(mesh)
    connectivity = cell_connectivity(mesh)
    isactivecell = active_cells(coeffs,poly)
    isactiveface = active_faces(coeffs,poly,isactivecell)
    cell2elid = number_elements(isactivecell)
    face2hid = number_face_hybrid_elements(isactiveface,connectivity)
    hid = maximum(face2hid)+1
    interface2hid = number_interface_hybrid_elements(isactivecell,hid)
    return DGMesh(domain,connectivity,isactivecell,isactiveface,
        cell2elid,face2hid,interface2hid)
end

function cell_domain(mesh::UniformMesh{dim,T}) where {dim,T}
    domain = Vector{IntervalBox{dim,T}}(undef,mesh.total_number_of_elements)
    for idx in 1:mesh.total_number_of_elements
        xL,xR = CartesianMesh.element(mesh,idx)
        domain[idx] = IntervalBox(xL,xR)
    end
    return domain
end

function cell_connectivity(mesh)
    nfaces = CartesianMesh.faces_per_cell(mesh)
    ncells = mesh.total_number_of_elements
    connectivity = Matrix{Tuple{Int,Int}}(undef,nfaces,ncells)

    for cellid in 1:ncells
        nbrcellids = CartesianMesh.neighbors(mesh,cellid)
        nbrfaceids = [nc == 0 ? 0 : neighbor_faceid(faceid) for (faceid,nc) in enumerate(nbrcellids)]
        nbrcellandface = collect(zip(nbrcellids,nbrfaceids))
        connectivity[:,cellid] .= nbrcellandface
    end
    return connectivity
end

function active_cells!(isactivecell,coeffs,
    poly::InterpolatingPolynomial{1,NF,B}) where
    {NF,B<:TensorProductBasis{dim}} where {dim}

    nf,ncells = size(coeffs)

    @assert nf == NF
    @assert size(isactivecell) == (2,ncells)


    xL,xR = reference_cell(dim)
    box = IntervalBox(xL,xR)
    fill!(isactivecell,false)

    for idx in 1:ncells
        update!(poly,coeffs[:,idx])
        s = sign(poly,box)
        if s == 1
            isactivecell[1,idx] = true
        elseif s == -1
            isactivecell[2,idx] = true
        elseif s == 0
            isactivecell[[1,2],idx] .= true
        else
            throw(ArgumentError("Expected s ∈ {-1,0,+1}, got s = $s"))
        end
    end
end

function active_cells(coeffs,poly)
    nf,ncells = size(coeffs)
    isactivecell = zeros(Bool,2,ncells)
    active_cells!(isactivecell,coeffs,poly)
    return isactivecell
end

function update_active_faces!(isactiveface,faceid,s,cellid)
    if s == 1
        isactiveface[faceid,1,cellid] = true
    elseif s == -1
        isactiveface[faceid,2,cellid] = true
    elseif s == 0
        isactiveface[faceid,1,cellid] = true
        isactiveface[faceid,2,cellid] = true
    else
        throw(ArgumentError("Expected s ∈ {-1,0,+1}, got s = $s"))
    end
end

function active_faces!(isactiveface,coeffs,poly,isactivecell)

    nf,ncells = size(coeffs)

    @assert size(isactivecell) == (2,ncells)
    @assert size(isactiveface) == (4,2,ncells)


    fill!(isactiveface,false)
    xL,xR = reference_cell(1)

    box = IntervalBox(xL,xR)

    for idx in 1:ncells
        if isactivecell[1,idx] && !isactivecell[2,idx]
            isactiveface[:,1,idx] .= true
        elseif !isactivecell[1,idx] && isactivecell[2,idx]
            isactiveface[:,2,idx] .= true
        elseif isactivecell[1,idx] && isactivecell[2,idx]

            update!(poly,coeffs[:,idx])
            funcs = restrict_on_faces(poly,xL[1],xR[1])
            facesigns = [sign(f,box) for f in funcs]

            for (faceid,fs) in enumerate(facesigns)
                update_active_faces!(isactiveface,faceid,fs,idx)
            end

        end
    end
end

function active_faces(coeffs,poly,isactivecell)
    nf,ncells = size(coeffs)
    isactiveface = zeros(Bool,4,2,ncells)
    active_faces!(isactiveface,coeffs,poly,isactivecell)
    return isactiveface
end

function number_elements!(cell2elid,isactivecell)

    nphase,ncells = size(isactivecell)
    @assert nphase == 2
    @assert size(cell2elid) == (2,ncells)

    fill!(cell2elid,0)
    elid = 1
    for idx in 1:ncells
        for phase = 1:2
            if isactivecell[phase,idx]
                cell2elid[phase,idx] = elid
                elid += 1
            end
        end
    end
end

function number_elements(isactivecell)
    nphase,ncells = size(isactivecell)
    cell2elid = zeros(Int,2,ncells)
    number_elements!(cell2elid,isactivecell)
    return cell2elid
end

function number_face_hybrid_elements!(face2hid,isactiveface,connectivity)

    nface,ncells = size(connectivity)
    @assert nface == 4
    @assert size(face2hid) == (4,2,ncells)
    @assert size(isactiveface) == (4,2,ncells)

    fill!(face2hid,0)
    hid = 1

    for idx in 1:ncells
        for phase in 1:2
            for faceid in 1:4
                if isactiveface[faceid,phase,idx] && face2hid[faceid,phase,idx] == 0
                    face2hid[faceid,phase,idx] = hid
                    nbr,nbrfaceid = connectivity[faceid,idx]
                    if nbr != 0
                        @assert face2hid[nbrfaceid,phase,nbr] == 0
                        face2hid[nbrfaceid,phase,nbr] = hid
                    end
                    hid += 1
                end
            end
        end
    end
end

function number_face_hybrid_elements(isactiveface,connectivity)
    face2hid = similar(isactiveface,Int)
    number_face_hybrid_elements!(face2hid,isactiveface,connectivity)
    return face2hid
end

function number_interface_hybrid_elements!(interface2hid,isactivecell,hid)

    nphase,ncells = size(isactivecell)
    @assert nphase == 2
    @assert size(interface2hid) == (2,ncells)
    @assert hid > 0

    fill!(interface2hid,0)

    for cellid in 1:ncells
        if isactivecell[1,cellid] && isactivecell[2,cellid]
            interface2hid[1,cellid] = hid
            hid += 1
            interface2hid[2,cellid] = hid
            hid += 1
        end
    end

end

function number_interface_hybrid_elements(isactivecell,hid)
    interface2hid = similar(isactivecell,Int)
    number_interface_hybrid_elements!(interface2hid,isactivecell,hid)
    return interface2hid
end
