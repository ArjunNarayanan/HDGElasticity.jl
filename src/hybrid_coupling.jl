function HHop(sbasis,squad,cellmap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    NHF = number_of_basis_functions(sbasis)

    cell = reference_cell(dim)
    jac = jacobian(cellmap,cell)

    nfaces = length(jac)
    ndofs = dim*NHF
    HH = [zeros(ndofs,ndofs) for i in 1:nfaces]

    for faceid in 1:nfaces
        update_mass_matrix!(HH[faceid],sbasis,squad,dim,stabilization*jac[faceid])
    end

    return HH
end

function HHop!(HH,sbasis,squad,components::M,scale,
    detjac,nudofs,NHF) where {M<:AbstractMatrix}

    NQ = length(squad)
    @assert size(components)[2] == NQ
    @assert size(HH) == (nudofs*NHF,nudofs*NHF)

    for (idx,(p,w)) in enumerate(squad)
        t = components[:,idx]
        projector = t*t'
        svals = sbasis(p)
        NP = make_row_matrix(svals,projector)
        NI = interpolation_matrix(svals,nudofs)

        HH .+= scale*NP'*NI*detjac*w
    end

end

function HHop!(HH,sbasis,squad,components::V,scale,
    detjac,nudofs,NHF) where {V<:AbstractVector}

    NQ = length(squad)
    extcomp = repeat(components,inner=(1,NQ))

    HHop!(HH,sbasis,squad,extcomp,scale,detjac,nudofs,NHF)

end

function HHop(sbasis,squad,components,faceid,cellmap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    NHF = number_of_basis_functions(sbasis)

    cell = reference_cell(dim)
    jac = jacobian(cellmap,cell)

    HH = zeros(dim*NHF,dim*NHF)

    HHop!(HH,sbasis,squad,components,stabilization,jac[faceid],dim,NHF)
    return HH

end

function HHop_on_active_faces(sbasis,facequads,isactiveface,cellmap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    NHF = number_of_basis_functions(sbasis)

    cell = reference_cell(dim)
    jac = jacobian(cellmap,cell)

    nfaces = length(jac)
    @assert length(facequads) == nfaces
    @assert length(isactiveface) == nfaces

    ndofs = dim*NHF
    HH = [zeros(ndofs,ndofs) for i in 1:nfaces]

    for faceid in 1:nfaces
        if isactiveface[faceid]
            update_mass_matrix!(HH[faceid],sbasis,facequads[faceid],
                dim,stabilization*jac[faceid])
        end
    end
    return HH
end

function HHop_on_interface(sbasis,iquad,normals,imap,cellmap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    NHF = number_of_basis_functions(sbasis)

    cell = reference_cell(dim)

    ndofs = dim*NHF
    HH = zeros(ndofs,ndofs)

    scale = scale_area(cellmap,normals)

    for (idx,(p,w)) in enumerate(iquad)
        vals = sbasis(p)
        N = interpolation_matrix(vals,dim)
        detjac = determinant_jacobian(imap,p)
        HH .+= N'*N*detjac*scale[idx]*w
    end
    return HH
end
