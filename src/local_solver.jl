function compute_local_solver_on_cells(dgmesh,ufs,D1,D2,cellmap,stabilization)

    ncells = length(dgmesh.domain)
    localsolver = Matrix{LocalOperator}(undef,2,ncells)

    uniformop1 = LocalOperator(ufs.vbasis,ufs.vtpq,ufs.ftpq,D1,
        cellmap,stabilization)
    uniformop2 = LocalOperator(ufs.vbasis,ufs.vtpq,ufs.ftpq,D2,
        cellmap,stabilization)

    isactivecell = dgmesh.isactivecell
    
    for cellid in 1:ncells
        if isactivecell[1,cellid] && !isactivecell[2,cellid]
            localsolver[1,cellid] = uniformop1
        elseif !isactivecell[1,cellid] && isactivecell[2,cellid]
            localsolver[2,cellid] = uniformop2
        elseif isactivecell[1,cellid] && isactivecell[2,cellid]
            localsolver[1,cellid] = LocalOperator(ufs.vbasis,
                ufs.vquads[1,cellid],view(ufs.fquads,:,1,cellid),
                view(dgmesh.isactiveface,:,1,cellid),D1,cellmap,stabilization)

            localsolver[2,cellid] = LocalOperator(ufs.vbasis,
                ufs.vquads[2,cellid],view(ufs.fquads,:,2,cellid),
                view(dgmesh.isactiveface,:,2,cellid),D2,cellmap,stabilization)
        end
    end

    return localsolver
end
