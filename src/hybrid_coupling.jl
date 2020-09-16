function HHop(sbasis,facequad,facescale)
    sdim = dimension(sbasis)
    vdim = sdim + 1
    NHF = number_of_basis_functions(sbasis)
    HH = zeros(vdim*NHF,vdim*NHF)
    update_mass_matrix!(HH,sbasis,facequad,facescale,vdim)
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

function HHop!(HH,sbasis,iquad,imap::InterpolatingPolynomial,scale,nhdofs)

    NQ = length(iquad)
    @assert length(scale) == NQ

    for (idx,(p,w)) in enumerate(iquad)
        vals = sbasis(p)
        detjac = determinant_jacobian(imap,p)
        N = interpolation_matrix(vals,nhdofs)
        HH .+= N'*N*detjac*scale[idx]*w
    end
end

function HHop(sbasis,iquad,imap,inormals,cellmap)

    facedim = dimension(sbasis)
    dim = facedim + 1
    NHF = number_of_basis_functions(sbasis)
    HH = zeros(dim*NHF,dim*NHF)

    facescale = scale_area(cellmap,inormals)
    HHop!(HH,sbasis,iquad,imap,facescale,dim)
    return HH
end

function HHop!(HH,sbasis,iquad,imap,components,scale,nhdofs)

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

function HHop(sbasis,iquad,imap,inormals,components,cellmap)

    facedim = dimension(sbasis)
    dim = facedim + 1
    NHF = number_of_basis_functions(sbasis)
    HH = zeros(dim*NHF,dim*NHF)

    facescale = scale_area(cellmap,inormals)
    HHop!(HH,sbasis,iquad,imap,components,facescale,dim)
    return HH
end
