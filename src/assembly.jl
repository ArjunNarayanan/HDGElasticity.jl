struct SystemMatrix{T,Z}
    rows::Vector{Z}
    cols::Vector{Z}
    vals::Vector{T}
    function SystemMatrix(rows::Vector{Z},cols::Vector{Z},
        vals::Vector{T}) where {T<:Real,Z<:Integer}

        @assert length(rows) == length(cols)
        @assert length(cols) == length(vals)

        new{T,Z}(rows,cols,vals)
    end
end

function SystemMatrix()
    Z = default_integer_type()
    R = default_float_type()
    rows = Z[]
    cols = Z[]
    vals = R[]
    return SystemMatrix(rows,cols,vals)
end

function assemble!(matrix::SystemMatrix,rows,cols,vals)
    @assert length(rows) == length(cols)
    @assert length(cols) == length(vals)

    append!(matrix.rows,rows)
    append!(matrix.cols,cols)
    append!(matrix.vals,vals)
end

struct SystemRHS{T,Z}
    rows::Vector{Z}
    vals::Vector{T}
    function SystemRHS(rows::Vector{Z},vals::Vector{T}) where {Z<:Integer,T<:Real}
        @assert length(rows) == length(vals)
        new{T,Z}(rows,vals)
    end
end

function SystemRHS()
    Z = default_integer_type()
    R = default_float_type()
    rows = Z[]
    vals = R[]
    return SystemRHS(rows,vals)
end

function assemble!(rhs::SystemRHS,rows,vals)
    @assert length(rows) == length(vals)
    append!(rhs.rows,rows)
    append!(rhs.vals,vals)
end

function element_dof_start(elid,dofsperelement)
    return (elid-1)*dofsperelement+1
end

function element_dof_stop(elid,dofsperelement)
    return elid*dofsperelement
end

function element_dofs(elid,dofsperelement)
    start = element_dof_start(elid,dofsperelement)
    stop = element_dof_stop(elid,dofsperelement)
    return start:stop
end

function operator_dofs(row_dofs,col_dofs)
    lr = length(row_dofs)
    lc = length(col_dofs)
    rows = repeat(row_dofs,outer=lc)
    cols = repeat(col_dofs,inner=lr)
    return rows,cols
end

function assemble!(system_matrix::SystemMatrix,vals,rowelids,
    colelids,dofsperelement)

    rowdofs = vcat([element_dofs(r,dofsperelement) for r in rowelids]...)
    coldofs = vcat([element_dofs(c,dofsperelement) for c in colelids]...)

    oprows,opcols = operator_dofs(rowdofs,coldofs)
    assemble!(system_matrix,oprows,opcols,vals)
end

function assemble!(system_rhs::SystemRHS,vals,rowelid,dofsperelement)
    rowdofs = element_dofs(rowelid,dofsperelement)
    assemble!(system_rhs,rowdofs,vals)
end

function assemble_displacement_face!(system_matrix::SystemMatrix,sbasis,
    facequad,facescale,elid,dofsperelement)

    HH = HHop(sbasis,facequad,facescale)
    assemble!(system_matrix,vec(HH),elid,elid,dofsperelement)
end

function assemble_displacement_face!(system_matrix::SystemMatrix,
    dgmesh::DGMesh,ufs::UniformFunctionSpace,phaseid,cellid,
    faceid,rowelid,dofsperelement)

    sbasis = ufs.sbasis
    facequad = ufs.fquads[phaseid,cellid][faceid]
    facescale = dgmesh.facescale[faceid]
    assemble_displacement_face!(system_matrix,sbasis,facequad,
        facescale,rowelid,dofsperelement)
end


function assemble_traction_face!(system_matrix::SystemMatrix,HL,iLLxLH,HH,
    rowelid,colelids,dofsperelement)

    tractionop = HL*iLLxLH
    assemble!(system_matrix,tractionop,rowelid,colelids,dofsperelement)
    assemble!(system_matrix,-HH,rowelid,rowelid,dofsperelement)
end

function assemble_traction_face!(system_matrix::SystemMatrix,sbasis,facequad,
    facescale,stabilization,HL,iLLxLH,rowelid,colelids,dofsperelement)

    HH = stabilization*HHop(sbasis,facequad,facescale)
    assemble_traction_face!(system_matrix,HL,iLLxLH,HH,rowelid,
        colelids,dofsperelement)
end

function assemble_traction_face!(system_rhs::SystemRHS,sbasis,facequad,
    facescale,rhsval,rowelid,dofsperelement)

    rhs = -facescale*linear_form(rhsval,sbasis,facequad)
    assemble!(system_rhs,rhs,rowelid,dofsperelement)
end

function assemble_traction_face!(system_rhs::SystemRHS,dgmesh::DGMesh,
    ufs::UniformFunctionSpace,rhsval,facetohelid,phaseid,cellid,faceid)

    rowelid = facetohelid[phaseid,cellid][faceid]
    facequad = ufs.fquads[phaseid,cellid][faceid]
    facescale = dgmesh.facescale[faceid]
    dofsperelement = ufs.dofsperelement
    assemble_traction_face!(system_rhs,ufs.sbasis,facequad,facescale,
        rhsval,rowelid,dofsperelement)
end

function assemble_traction_face!(system_matrix::SystemMatrix,
    dgmesh::DGMesh,ufs::UniformFunctionSpace,cellsolvers::CellSolvers,
    phaseid,cellid,faceid,rowelid,colelids,dofsperelement)

    sbasis = ufs.sbasis
    facequad = ufs.fquads[phaseid,cellid][faceid]
    facescale = dgmesh.facescale[faceid]
    cellsolver = cellsolvers[phaseid,cellid]
    stabilization = cellsolvers.stabilization
    iLLxLH = cellsolver.iLLxLH
    facetosolverid = cellsolver.facetosolverid
    HL = cellsolver.fLH[facetosolverid[faceid]]'
    assemble_traction_face!(system_matrix,sbasis,facequad,
        facescale,stabilization,HL,iLLxLH,rowelid,colelids,dofsperelement)
end

function assemble_traction_coherent_interface!(system_matrix::SystemMatrix,
    HL1,iLLxLH1,HL2,iLLxLH2,HH,hid1,colelids1,hid2,colelids2,dofsperelement)

    op1 = HL1*iLLxLH1
    op2 = HL2*iLLxLH2

    assemble!(system_matrix,op1,hid1,colelids1,dofsperelement)
    assemble!(system_matrix,-HH,hid1,hid1,dofsperelement)
    assemble!(system_matrix,op2,hid1,colelids2,dofsperelement)
    assemble!(system_matrix,-HH,hid1,hid2,dofsperelement)

    assemble!(system_matrix,op1,hid2,colelids1,dofsperelement)
    assemble!(system_matrix,-HH,hid2,hid1,dofsperelement)
    assemble!(system_matrix,op2,hid2,colelids2,dofsperelement)
    assemble!(system_matrix,-HH,hid2,hid2,dofsperelement)
end

function assemble_displacement_coherent_interface!(system_matrix::SystemMatrix,
    HH,rowelid,colelid,dofsperelement)

    vals = vec(HH)
    assemble!(system_matrix,vals,rowelid,rowelid,dofsperelement)
    assemble!(system_matrix,-vals,rowelid,colelid,dofsperelement)
    assemble!(system_matrix,-vals,colelid,rowelid,dofsperelement)
    assemble!(system_matrix,vals,colelid,colelid,dofsperelement)
end


function assemble_coherent_interface!(system_matrix::SystemMatrix,
    dgmesh::DGMesh,ufs::UniformFunctionSpace,cellsolvers::CellSolvers,
    cellid,hid1,colelids1,hid2,colelids2,dofsperelement)

    sbasis = ufs.sbasis
    iquad = ufs.iquad
    imap = ufs.imap
    inormals = ufs.inormals[cellid]
    stabilization = cellsolvers.stabilization
    update!(imap,ufs.icoeffs[cellid])
    HH = HHop(sbasis,iquad,imap,inormals,dgmesh.cellmap)

    cs1 = cellsolvers[1,cellid]
    iLLxLH1 = cs1.iLLxLH
    facetosolverid = cs1.facetosolverid
    HL1 = cs1.fLH[facetosolverid[5]]'

    cs2 = cellsolvers[2,cellid]
    iLLxLH2 = cs2.iLLxLH
    facetosolverid = cs2.facetosolverid
    HL2 = cs2.fLH[facetosolverid[5]]'

    assemble_displacement_coherent_interface!(system_matrix,HH,
        hid1,hid2,dofsperelement)
    assemble_traction_coherent_interface!(system_matrix,
        HL1,iLLxLH1,HL2,iLLxLH2,stabilization*HH,hid1,colelids1,
        hid2,colelids2,dofsperelement)
end

function assemble_mixed_face!(system_matrix::SystemMatrix,vbasis,sbasis,
    facequad,facemap,facenormal,dcomp,tcomp,Dhalf,stabilization,facescale,
    iLLxLH,rowelid,colelids,dofsperelement)

    HHd = HHop(sbasis,facequad,dcomp,facescale)
    assemble!(system_matrix,vec(HHd),rowelid,rowelid,dofsperelement)

    HLt = hybrid_local_operator_traction_components(sbasis,vbasis,facequad,
        facemap,facenormal,tcomp,Dhalf,stabilization,facescale)
    HHt = stabilization*HHop(sbasis,facequad,tcomp,facescale)
    assemble_traction_face!(system_matrix,HLt,iLLxLH,HHt,rowelid,colelids,
        dofsperelement)
end

function assemble_mixed_face!(system_matrix::SystemMatrix,
    dgmesh::DGMesh,ufs::UniformFunctionSpace,cellsolvers::CellSolvers,
    phaseid,cellid,faceid,dcomp,tcomp,rowelid,colelids,dofsperelement)

    vbasis = ufs.vbasis
    sbasis = ufs.sbasis
    facequad = ufs.fquads[phaseid,cellid][faceid]
    facemap = dgmesh.facemaps[faceid]
    facenormal = ufs.fnormals[faceid]
    facescale = dgmesh.facescale[faceid]
    iLLxLH = cellsolvers[phaseid,cellid].iLLxLH
    D = cellsolvers.stiffness[phaseid]
    stabilization = cellsolvers.stabilization
    assemble_mixed_face!(system_matrix,vbasis,sbasis,facequad,facemap,
        facenormal,dcomp,tcomp,D,stabilization,facescale,iLLxLH,
        rowelid,colelids,dofsperelement)
end

function SparseArrays.sparse(system_matrix::SystemMatrix,ndofs)
    return sparse(system_matrix.rows,system_matrix.cols,system_matrix.vals,
        ndofs,ndofs)
end

function rhs(system_rhs::SystemRHS,ndofs)
    return Array(sparsevec(system_rhs.rows,system_rhs.vals,ndofs))
end

function assign_cell_hybrid_element_ids!(facetohelid,visited,
    isactiveface,helid,nfaces)

    for faceid in 1:nfaces
        if !visited[faceid] && isactiveface[faceid]
            facetohelid[faceid] = helid
            visited[faceid] = true
            helid += 1
        end
    end
    return helid
end

function assign_neighbor_cell_hybrid_ids!(facetohelid,visited,phaseid,cellid,
    connectivity,nfaces)

    for faceid = 1:nfaces
        nbrcellid,nbrfaceid = connectivity[cellid][faceid]
        if nbrcellid != 0 && !visited[phaseid,nbrcellid][nbrfaceid]
            facetohelid[phaseid,nbrcellid][nbrfaceid] = facetohelid[phaseid,cellid][faceid]
            visited[phaseid,nbrcellid][nbrfaceid] = true
        end
    end
end

function cell_hybrid_element_ids(cellsign,isactiveface,connectivity,nfaces)
    ncells = length(cellsign)
    facetohelid = [zeros(Int,nfaces) for i = 1:2, j = 1:ncells]
    interfacehelid = zeros(Int,2,ncells)
    visited = [zeros(Bool,nfaces) for i = 1:2, j = 1:ncells]
    helid = 1
    for cellid = 1:ncells
        s = cellsign[cellid]
        if s == 1
            helid = assign_cell_hybrid_element_ids!(facetohelid[1,cellid],
                visited[1,cellid],isactiveface[1,cellid],helid,nfaces)
            assign_neighbor_cell_hybrid_ids!(facetohelid,visited,1,cellid,
                connectivity,nfaces)
        elseif s == -1
            helid = assign_cell_hybrid_element_ids!(facetohelid[2,cellid],
                visited[2,cellid],isactiveface[2,cellid],helid,nfaces)
            assign_neighbor_cell_hybrid_ids!(facetohelid,visited,2,cellid,
                connectivity,nfaces)
        elseif s == 0
            helid = assign_cell_hybrid_element_ids!(facetohelid[1,cellid],
                visited[1,cellid],isactiveface[1,cellid],helid,nfaces)
            assign_neighbor_cell_hybrid_ids!(facetohelid,visited,1,cellid,
                connectivity,nfaces)
            interfacehelid[1,cellid] = helid
            helid += 1
            helid = assign_cell_hybrid_element_ids!(facetohelid[2,cellid],
                visited[2,cellid],isactiveface[2,cellid],helid,nfaces)
            assign_neighbor_cell_hybrid_ids!(facetohelid,visited,2,cellid,
                connectivity,nfaces)
            interfacehelid[2,cellid] = helid
            helid += 1
        else
            error("Expected cellsign ∈ {-1,0,1}, got cellsign = $s")
        end
    end
    return facetohelid,interfacehelid,helid
end

struct HybridElementNumbering
    facetohelid
    interfacehelid
    number_of_hybrid_elements
end

function HybridElementNumbering(dgmesh,ufs)
    dim = dimension(dgmesh)
    nfaces = number_of_faces(dim)
    facetohelid,interfacehelid,helidstop = cell_hybrid_element_ids(
        dgmesh.cellsign,ufs.isactiveface,dgmesh.connectivity,nfaces
    )
    number_of_hybrid_elements = helidstop - 1
    return HybridElementNumbering(facetohelid,interfacehelid,
        number_of_hybrid_elements)
end

function interior_operator(ls)
    return ls.LH'*ls.iLLxLH
end

function assemble_uniform_interior_faces!(system_matrix,cellsolvers,
    cellsign,isinteriorcell,facetohelid,dofsperelement)

    vals1 = vec(interior_operator(cellsolvers[1]))
    vals2 = vec(interior_operator(cellsolvers[2]))
    ncells = length(cellsign)
    cellids = findall(isinteriorcell)

    for cellid in cellids
        s = cellsign[cellid]
        if s == 1
            helids = facetohelid[1,cellid]
            @assert isnothing(findfirst(x->x==0,helids))
            assemble!(system_matrix,vals1,helids,helids,dofsperelement)
        elseif s == -1
            helids = facetohelid[2,cellid]
            @assert isnothing(findfirst(x->x==0,helids))
            assemble!(system_matrix,vals2,helids,helids,dofsperelement)
        end
    end
end

function assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,interfacehelid,phaseid,cellid)

    dofsperelement = ufs.dofsperelement
    fhelid = facetohelid[phaseid,cellid]
    ihelid = interfacehelid[phaseid,cellid]
    colelids = filter(x->x!=0,fhelid)
    push!(colelids,ihelid)

    for (faceid,helid) in enumerate(fhelid)
        if helid != 0
            assemble_traction_face!(system_matrix,
                dgmesh,ufs,cellsolvers,phaseid,cellid,faceid,
                helid,colelids,dofsperelement)
        end
    end
end

function assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
    cellsolvers,facetohelid,interfacehelid)

    cellsign = dgmesh.cellsign
    isinteriorcell = dgmesh.isinteriorcell
    cellids = findall([x==0 for x in cellsign] .& isinteriorcell)

    for cellid in cellids
        assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
            cellsolvers,facetohelid,interfacehelid,1,cellid)
        assemble_cut_interior_faces!(system_matrix,dgmesh,ufs,
            cellsolvers,facetohelid,interfacehelid,2,cellid)
    end
end

function assemble_boundary_face!(system_matrix,dgmesh,ufs,cellsolvers,
    phaseid,cellid,faceid,midpoint,rowelid,colelids,bc_classifier,bc_data)

    dofsperelement = ufs.dofsperelement
    bdrytype = bc_classifier(midpoint)
    if bdrytype == :displacement
        assemble_displacement_face!(system_matrix,dgmesh,ufs,
            phaseid,cellid,faceid,rowelid,dofsperelement)
    elseif bdrytype == :traction
        assemble_traction_face!(system_matrix,dgmesh,ufs,
            cellsolvers,phaseid,cellid,faceid,rowelid,
            colelids,dofsperelement)
    elseif bdrytype == :mixed
        dcomp,tcomp = bc_data[:mixed_components](midpoint)
        assemble_mixed_face!(system_matrix,dgmesh,ufs,
            cellsolvers,phaseid,cellid,faceid,dcomp,tcomp,
            rowelid,colelids,dofsperelement)
    else
        error("Unexpected boundary type encountered")
    end
end

function assemble_uniform_boundary_faces!(system_matrix,dgmesh,
    ufs,cellsolvers,facetohelid,phaseid,cellid,bc_classifier,bc_data)

    cellmap = CellMap(dgmesh.domain[cellid])
    midpoints = face_midpoints(cellmap)
    colelids = facetohelid[phaseid,cellid]
    connectivity = dgmesh.connectivity[cellid]
    dofsperelement = ufs.dofsperelement

    for (faceid,xm) in enumerate(midpoints)
        rowelid = colelids[faceid]
        if connectivity[faceid][1] != 0
            assemble_traction_face!(system_matrix,dgmesh,ufs,cellsolvers,
                phaseid,cellid,faceid,rowelid,colelids,dofsperelement)
        else
            assemble_boundary_face!(system_matrix,dgmesh,ufs,cellsolvers,
                phaseid,cellid,faceid,xm,rowelid,colelids,bc_classifier,
                bc_data)
        end
    end
end

function assemble_uniform_boundary_faces!(system_matrix,dgmesh,
    ufs,cellsolvers,facetohelid,bc_classifier,bc_data)

    cellids = findall(.!dgmesh.isinteriorcell)
    cellsign = dgmesh.cellsign
    for cellid in cellids
        if cellsign[cellid] == 1
            assemble_uniform_boundary_faces!(system_matrix,dgmesh,
            ufs,cellsolvers,facetohelid,1,cellid,bc_classifier,bc_data)
        elseif cellsign[cellid] == -1
            assemble_uniform_boundary_faces!(system_matrix,dgmesh,
            ufs,cellsolvers,facetohelid,2,cellid,bc_classifier,bc_data)
        end
    end
end

function assemble_cut_boundary_faces!(system_matrix,dgmesh,
    ufs,cellsolvers,facetohelid,interfacehelid,phaseid,cellid,
    bc_classifier,bc_data)

    dofsperelement = ufs.dofsperelement
    fhelid = facetohelid[phaseid,cellid]
    ihelid = interfacehelid[phaseid,cellid]
    colelids = filter(x->x!=0,fhelid)
    push!(colelids,ihelid)
    cellmap = CellMap(dgmesh.domain[cellid])
    xm = face_midpoints(cellmap)

    for (faceid,helid) in enumerate(fhelid)
        if helid != 0

        end
    end
end
