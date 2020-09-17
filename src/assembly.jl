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
