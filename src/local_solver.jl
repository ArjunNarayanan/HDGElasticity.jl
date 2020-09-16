struct LocalSolverComponents
    LL
    fLH
    LH
    iLLxLH
    facetosolverid
    function LocalSolverComponents(LL,fLH,facetosolverid)
        LH = hcat(fLH...)
        iLLxLH = LL\LH
        new(LL,fLH,LH,iLLxLH,facetosolverid)
    end
end

struct CellSolvers
    localsolvers::Vector{LocalSolverComponents}
    celltosolverid::Matrix{Int}
end

function Base.getindex(CS::CellSolvers,phaseid,cellid)
    return CS.localsolvers[CS.celltosolverid[phaseid,cellid]]
end

function face_to_solverid(facequads)
    nfaces = length(facequads)
    isactiveface = [length(fq) > 0 ? true : false for fq in facequads]
    solverid = zeros(Int,nfaces)
    counter = 1
    for faceid in 1:nfaces
        if isactiveface[faceid]
            solverid[faceid] = counter
            counter += 1
        end
    end
    return solverid
end

function LocalSolverComponents(vbasis,vquad,sbasis,facequads,facemaps,normals,
    Dhalf,stabilization,cellmap)

    LL = local_operator(vbasis,vquad,facequads,facemaps,Dhalf,
        stabilization,cellmap)
    fLH = local_hybrid_operator(vbasis,sbasis,facequads,facemaps,normals,Dhalf,
        stabilization,cellmap)
    facetosolverid = face_to_solverid(facequads)
    return LocalSolverComponents(LL,fLH,facetosolverid)
end

function LocalSolverComponents(vbasis,vquad,sbasis,facequads,facemaps,normals,
    iquad,imap,inormals,Dhalf,stabilization,cellmap)

    LL = local_operator(vbasis,vquad,facequads,facemaps,iquad,imap,inormals,
        Dhalf,stabilization,cellmap)
    fLH = local_hybrid_operator(vbasis,sbasis,facequads,facemaps,normals,
        Dhalf,stabilization,cellmap)
    iLH = local_hybrid_operator_on_interface(vbasis,sbasis,iquad,imap,inormals,
        Dhalf,stabilization,cellmap)
    push!(fLH,iLH)
    facetosolverid = face_to_solverid(facequads)
    push!(facetosolverid,length(fLH))
    return LocalSolverComponents(LL,fLH,facetosolverid)
end

function cell_to_solver_index(cellsign)
    ncells = length(cellsign)
    solveridx = zeros(Int,2,ncells)
    counter = 3
    for cellid in 1:ncells
        s = cellsign[cellid]
        if s == +1
            solveridx[1,cellid] = 1
        elseif s == -1
            solveridx[2,cellid] = 2
        elseif s == 0
            solveridx[1,cellid] = counter
            counter += 1
            solveridx[2,cellid] = counter
            counter += 1
        else
            str = "Expected cellsign ∈ {-1,0,+1}, got s = $s"
            error(str)
        end
    end
    return solveridx
end

function CellSolvers(dgmesh,ufs,D1,D2,stabilization)

    ncells = number_of_cells(dgmesh)
    @assert ncells > 0
    cellmap = CellMap(dgmesh.domain[1])

    uniformop1 = LocalSolverComponents(ufs.vbasis,ufs.vtpq,ufs.sbasis,ufs.ftpq,
        dgmesh.facemaps,ufs.fnormals,D1,stabilization,cellmap)
    uniformop2 = LocalSolverComponents(ufs.vbasis,ufs.vtpq,ufs.sbasis,ufs.ftpq,
        dgmesh.facemaps,ufs.fnormals,D2,stabilization,cellmap)

    solver_components = [uniformop1,uniformop2]

    cellsign = dgmesh.cellsign

    for cellid in 1:ncells
        s = cellsign[cellid]
        if s == 0
            update!(ufs.imap,ufs.icoeffs[cellid])
            negativenormals = ufs.inormals[cellid]
            positivenormals = -negativenormals

            cutop1 = LocalSolverComponents(ufs.vbasis,
                ufs.vquads[1,cellid],ufs.sbasis,ufs.fquads[1,cellid],
                dgmesh.facemaps,ufs.fnormals,ufs.iquad,ufs.imap,positivenormals,
                D1,stabilization,cellmap)

            push!(solver_components,cutop1)

            cutop2 = LocalSolverComponents(ufs.vbasis,
                ufs.vquads[2,cellid],ufs.sbasis,ufs.fquads[2,cellid],
                dgmesh.facemaps,ufs.fnormals,ufs.iquad,ufs.imap,negativenormals,
                D2,stabilization,cellmap)

            push!(solver_components,cutop2)
        end
    end
    celltosolverid = cell_to_solver_index(cellsign)
    return CellSolvers(solver_components,celltosolverid)
end
