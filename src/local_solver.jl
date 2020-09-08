function local_operator_on_cells(dgmesh,ufs,D1,D2,cellmap,stabilization)

    ncells = number_of_cells(dgmesh)
    ft = default_float_type()

    localsolver = Matrix{Matrix{T}}(undef,2,ncells)

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

            update!(ufs.imap,ufs.icoeffs[:,cellid])
            negativenormals = ufs.inormals[:,:,cellid]
            positivenormals = -negativenormals

            localsolver[1,cellid] = LocalOperator(ufs.vbasis,
                ufs.vquads[1,cellid],view(ufs.fquads,:,1,cellid),
                view(dgmesh.isactiveface,:,1,cellid),ufs.iquad,positivenormals,
                ufs.imap,D1,cellmap,stabilization)

            localsolver[2,cellid] = LocalOperator(ufs.vbasis,
                ufs.vquads[2,cellid],view(ufs.fquads,:,2,cellid),
                view(dgmesh.isactiveface,:,2,cellid),ufs.iquad,negativenormals,
                ufs.imap,D2,cellmap,stabilization)
        end
    end

    return localsolver
end

function local_hybrid_operator_on_cells(dgmesh,ufs,D1,D2,
    cellmap,stabilization)

    nface,nphase,ncells = size(dgmesh.isactiveface)
    ft = default_float_type()

    lhc = Array{Matrix{ft}}(undef,nface,nphase,ncells)

    l1 = local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,
        D1,cellmap,stabilization)
    l2 = local_hybrid_operator(ufs.vbasis,ufs.sbasis,ufs.ftpq,
        D2,cellmap,stabilization)

    isactivecell = dgmesh.isactivecell
    for cellid in 1:ncells
        if isactivecell[1,cellid] && !isactivecell[2,cellid]
            lhc[:,1,cellid] = l1
        elseif !isactivecell[1,cellid] && isactivecell[2,cellid]
            lhc[:,2,cellid] = l2
        elseif isactivecell[1,cellid] && isactivecell[2,cellid]
            lhc[:,1,cellid] = local_hybrid_operator_on_active_faces(ufs.vbasis,
                ufs.sbasis,view(ufs.fquads,:,1,cellid),
                view(dgmesh.isactiveface,:,1,cellid),D1,cellmap,stabilization)

            lhc[:,2,cellid] = local_hybrid_operator_on_active_faces(ufs.vbasis,
                ufs.sbasis,view(ufs.fquads,:,2,cellid),
                view(dgmesh.isactiveface,:,2,cellid),D2,cellmap,stabilization)
        end
    end

    return lhc
end

function local_hybrid_operator_on_interfaces(dgmesh,ufs,D1,D2,
    cellmap,stabilization)

    ncells = length(dgmesh.domain)
    ft = default_float_type()

    lhci = Matrix{Matrix{ft}}(undef,2,ncells)

    isactivecell = dgmesh.isactivecell
    for cellid in 1:ncells

        if isactivecell[1,cellid] && isactivecell[2,cellid]
            negativenormals = ufs.inormals[:,:,cellid]
            positivenormals = -negativenormals

            update!(ufs.imap,ufs.icoeffs[:,cellid])
            lhci[1,cellid] = local_hybrid_operator_on_interface(ufs.vbasis,
                ufs.sbasis,ufs.iquad,positivenormals,D1,ufs.imap,cellmap,
                stabilization)
            lhci[2,cellid] = local_hybrid_operator_on_interface(ufs.vbasis,
                ufs.sbasis,ufs.iquad,negativenormals,D2,ufs.imap,cellmap,
                stabilization)
        end
    end
    return lhci
end

function hybrid_operator_on_cells(dgmesh,ufs,cellmap,stabilization)

    nface,nphase,ncells = size(dgmesh.isactiveface)
    ft = default_float_type()

    hhop = Array{Matrix{ft}}(undef,nface,nphase,ncells)

    uniformhop = HHop(ufs.sbasis,ufs.ftpq,cellmap,stabilization)

    isactivecell = dgmesh.isactivecell

    for cellid in 1:ncells
        if isactivecell[1,cellid] && !isactivecell[2,cellid]
            hhop[:,1,cellid] = uniformhop
        elseif !isactivecell[1,cellid] && isactivecell[2,cellid]
            hhop[:,2,cellid] = uniformhop
        elseif isactivecell[1,cellid] && isactivecell[2,cellid]
            hhop[:,1,cellid] = HHop_on_active_faces(ufs.sbasis,
                view(ufs.fquads,:,1,cellid),
                view(dgmesh.isactiveface,:,1,cellid),cellmap,stabilization)

            hhop[:,2,cellid] = HHop_on_active_faces(ufs.sbasis,
                view(ufs.fquads,:,2,cellid),
                view(dgmesh.isactiveface,:,2,cellid),cellmap,stabilization)
        end
    end
    return hhop
end
