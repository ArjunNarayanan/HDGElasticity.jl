function hybrid_operator(sbasis,facequads,stabilization,cellmap::CellMap)

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

function HHop(sbasis,squad,components,stabilization,facescale)
    return HHop(sbasis,squad,components,stabilization*facescale)
end

function HHop_on_interface!(HH,sbasis,iquad,imap,scale,nhdofs)

    NQ = length(iquad)
    @assert length(scale) == NQ

    for (idx,(p,w)) in enumerate(iquad)
        vals = sbasis(p)
        detjac = determinant_jacobian(imap,p)
        N = interpolation_matrix(vals,nhdofs)
        HH .+= N'*N*detjac*scale[idx]*w
    end
end

function hybrid_operator_on_interface(sbasis,iquad,imap,inormals,
    stabilization,cellmap)
    
    facedim = dimension(sbasis)
    dim = facedim + 1
    NHF = number_of_basis_functions(sbasis)
    HH = zeros(dim*NHF,dim*NHF)

    facescale = stabilization*scale_area(cellmap,inormals)
    HHop_on_interface!(HH,sbasis,iquad,imap,facescale,dim)
    return HH
end

function HHop_on_interface!(HH,sbasis,iquad,imap,components,scale,nhdofs)

    NQ = length(iquad)
    @assert length(scale) == NQ
    @assert size(components) == (nhdofs,NQ)

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

function hybrid_operator_on_interface(sbasis,iquad,imap,inormals,components,
    stabilization,cellmap)

    facedim = dimension(sbasis)
    dim = facedim + 1
    NHF = number_of_basis_functions(sbasis)
    HH = zeros(dim*NHF,dim*NHF)

    facescale = stabilization*scale_area(cellmap,inormals)
    HHop_on_interface!(HH,sbasis,iquad,imap,components,facescale,dim)
    return HH
end
