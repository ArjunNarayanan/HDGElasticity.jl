struct DGMesh{dim,T}
    domain::Vector{IntervalBox{dim,T}}
    connectivity::Matrix{Int}
    isactivecell::Matrix{Bool}
    isactiveface::Array{Bool,3}
    cell2elid::Matrix{Int}
    face2hid::Array{Int,3}
    interface2hid::Vector{Int}
    elids::UnitRange{Int}
    facehids::UnitRange{Int}
    interfacehids::UnitRange{Int}
    function DGMesh(domain::Vector{IntervalBox{dim,T}},connectivity::Matrix{Int},
        isactivecell::Matrix{Bool},isactiveface::Array{Bool,3},
        cell2elid::Matrix{Int},face2hid::Array{Int,3},
        interface2hid::Vector{Int}) where
        {dim,T}

        @assert dim == 2

        ncells = length(domain)
        @assert size(connectivity) == (4,ncells)
        @assert size(isactivecell) == (2,ncells)
        @assert size(isactiveface) == (4,2,ncells)
        @assert size(cell2elid) == (2,ncells)
        @assert size(face2hid) == (4,2,ncells)
        @assert length(interface2hid) == ncells

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
    connectivity = zeros(Int,4,mesh.total_number_of_elements)
    for idx in 1:mesh.total_number_of_elements
        connectivity[:,idx] .= CartesianMesh.neighbors(mesh,idx)
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
            fb(x) = poly(x,xL[1])
            fr(y) = poly(xR[1],y)
            ft(x) = poly(x,xR[1])
            fl(y) = poly(xL[1],y)

            sb = sign(fb,box)
            sr = sign(fr,box)
            st = sign(ft,box)
            sl = sign(fl,box)

            update_active_faces!(isactiveface,1,sb,idx)
            update_active_faces!(isactiveface,2,sr,idx)
            update_active_faces!(isactiveface,3,st,idx)
            update_active_faces!(isactiveface,4,sl,idx)

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

function neighbor_faceid(faceid)
    if faceid == 1
        return 3
    elseif faceid == 2
        return 4
    elseif faceid == 3
        return 1
    elseif faceid == 4
        return 2
    else
        throw(ArgumentError("Expected faceid ∈ {1,2,3,4}, got faceid = $faceid"))
    end
end

function number_face_hybrid_elements!(face2hid,isactiveface,connectivity)

    nface,ncells = size(connectivity)
    @assert nface == 4
    @assert size(face2hid) == (4,2,ncells)
    @assert size(isactiveface) == (4,2,ncells)

    fill!(face2hid,0)
    hid = 1

    for idx = 1:ncells
        for phase = 1:2
            for faceid = 1:4
                if isactiveface[faceid,phase,idx] && face2hid[faceid,phase,idx] == 0
                    face2hid[faceid,phase,idx] = hid
                    nbr = connectivity[faceid,idx]
                    if nbr != 0
                        nbrfaceid = neighbor_faceid(faceid)
                        face2hid[nbrfaceid,phase,nbr] = hid
                    end
                    hid += 1
                end
            end
        end
    end
end

function number_face_hybrid_elements(isactiveface,connectivity)
    nface,ncells = size(connectivity)
    face2hid = zeros(Int,4,2,ncells)
    number_face_hybrid_elements!(face2hid,isactiveface,connectivity)
    return face2hid
end

function number_interface_hybrid_elements!(interface2hid,isactivecell,hid)

    nphase,ncells = size(isactivecell)
    @assert nphase == 2
    @assert length(interface2hid) == ncells
    @assert hid > 0

    fill!(interface2hid,0)

    for idx in 1:ncells
        if isactivecell[1,idx] && isactivecell[2,idx]
            interface2hid[idx] = hid
            hid += 1
        end
    end

end

function number_interface_hybrid_elements(isactivecell,hid)
    nphase,ncells = size(isactivecell)
    interface2hid = zeros(Int,ncells)
    number_interface_hybrid_elements!(interface2hid,isactivecell,hid)
    return interface2hid
end
