function HHop(sbasis,facequads,cellmap::CellMap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    nfaces = number_of_faces(dim)
    @assert length(facequads) == nfaces

    NHF = number_of_basis_functions(sbasis)
    isactiveface = active_faces(facequads)

    jac = face_determinant_jacobian(cellmap)

    ndofs = dim*NHF
    HH = [zeros(ndofs,ndofs) for i in 1:count(isactiveface)]

    counter = 1
    for faceid in 1:nfaces
        if isactiveface[faceid]
            update_mass_matrix!(HH[counter],sbasis,facequads[faceid],
                stabilization*jac[faceid],dim)
            counter += 1
        end
    end
    return HH
end

function HHop!(HH,sbasis,squad,components,scale,nhdofs)

    NQ = length(squad)
    @assert length(components) == nhdofs
    projector = components*components'

    for (idx,(p,w)) in enumerate(squad)
        svals = sbasis(p)
        NP = make_row_matrix(svals,projector)
        NI = interpolation_matrix(svals,nhdofs)

        HH .+= scale*NP'*NI*w
    end
end

function HHop(sbasis,squad,components,scale)
    facedim = dimension(sbasis)
    dim = facedim + 1
    NHF = number_of_basis_functions(sbasis)
    HH = zeros(dim*NHF,dim*NHF)

    HHop!(HH,sbasis,squad,components,scale,dim)
    return HH
end

function HHop_on_interface!(HH,sbasis,iquad,normals,imap,cellmap,
    stabilization,nhdofs)

    NQ = length(iquad)
    @assert size(normals)[2] == NQ
    scale = stabilization*scale_area(cellmap,normals)

    for (idx,(p,w)) in enumerate(iquad)
        vals = sbasis(p)
        detjac = determinant_jacobian(imap,p)
        N = interpolation_matrix(vals,nhdofs)
        HH .+= N'*N*detjac*scale[idx]*w
    end
end

function HHop_on_interface!(HH,sbasis,iquad,normals,components,
    imap,cellmap,stabilization,nhdofs)

    NQ = length(iquad)
    @assert size(normals)[2] == NQ
    @assert size(components) == (nhdofs,NQ)

    scale = stabilization*scale_area(cellmap,normals)

    for (idx,(p,w)) in enumerate(iquad)
        t = components[:,idx]
        projector = t*t'
        vals = sbasis(p)
        NP = make_row_matrix(vals,projector)
        NI = interpolation_matrix(vals,nhdofs)
        detjac = determinant_jacobian(imap,p)
        HH .+= NP'*NI*detjac*scale[idx]*w
    end
    return HH
end
