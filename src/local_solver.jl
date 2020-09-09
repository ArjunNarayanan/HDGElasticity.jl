struct LocalSolver
    LLops
    LHops
    invLLxLH
    solveridx
    function LocalSolver(LLops,LHops,solveridx)
        @assert length(LLops) == length(LHops)
        invLLxLH = [LLops[i]\LHops[i] for i = 1:length(LLops)]
        new(LLops,LHops,invLLxLH,solveridx)
    end
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

function local_operator_on_cells(dgmesh,ufs,D1,D2,
    stabilization)

    ncells = number_of_cells(dgmesh)
    @assert ncells > 0
    cellmap = CellMap(dgmesh.domain[1])

    uniformop1 = local_operator(ufs.vbasis,ufs.vtpq,ufs.ftpq,dgmesh.facemaps,
        D1,stabilization,cellmap)
    uniformop2 = local_operator(ufs.vbasis,ufs.vtpq,ufs.ftpq,dgmesh.facemaps,
        D2,stabilization,cellmap)

    locops = [uniformop1,uniformop2]

    cellsign = dgmesh.cellsign

    for cellid in 1:ncells
        s = cellsign[cellid]
        if s == 0
            update!(ufs.imap,ufs.icoeffs[cellid])
            negativenormals = ufs.inormals[cellid]
            positivenormals = -negativenormals

            cutop1 = local_operator(ufs.vbasis,
                ufs.vquads[1,cellid],ufs.fquads[1,cellid],
                dgmesh.facemaps,ufs.iquad,positivenormals,
                ufs.imap,D1,stabilization,cellmap)

            push!(locops,cutop1)

            cutop2 = local_operator(ufs.vbasis,
                ufs.vquads[2,cellid],ufs.fquads[2,cellid],
                dgmesh.facemaps,ufs.iquad,negativenormals,
                ufs.imap,D2,stabilization,cellmap)

            push!(locops,cutop2)
        end
    end
    return locops
end

function local_hybrid_operator_on_cells(dgmesh,ufs,D1,D2,
    stabilization)

    ncells = number_of_cells(dgmesh)
    @assert ncells > 0
    cellmap = CellMap(dgmesh.domain[1])
    dim = dimension(dgmesh)
    nfaces = number_of_faces(dim)

    uniformop1 = local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,dgmesh.facemaps,
        ufs.fnormals,D1,stabilization,cellmap)
    uniformop2 = local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,dgmesh.facemaps,
        ufs.fnormals,D2,stabilization,cellmap)

    lochybops = [uniformop1,uniformop2]

    for cellid in 1:ncells
        s = dgmesh.cellsign[cellid]
        if s == 0

            update!(ufs.imap,ufs.icoeffs[cellid])
            negativenormals = ufs.inormals[cellid]
            positivenormals = -negativenormals

            cutfaceop1 = local_hybrid_operator(ufs.vbasis,ufs.sbasis,
                ufs.fquads[1,cellid],dgmesh.facemaps,ufs.fnormals,D1,
                stabilization,cellmap)
            iop1 = local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
                ufs.iquad,ufs.imap,positivenormals,D1,stabilization,cellmap)
            cutop1 = hcat(cutfaceop1,iop1)

            push!(lochybops,cutop1)

            cutfaceop2 = local_hybrid_operator(ufs.vbasis,ufs.sbasis,
                ufs.fquads[2,cellid],dgmesh.facemaps,ufs.fnormals,D2,
                stabilization,cellmap)
            iop2 = local_hybrid_operator_on_interface(ufs.vbasis,ufs.sbasis,
                ufs.iquad,ufs.imap,negativenormals,D2,stabilization,cellmap)
            cutop2 = hcat(cutfaceop2,iop2)

            push!(lochybops,cutop2)
        end
    end
    return lochybops
end
