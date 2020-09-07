function HHop(sbasis,facequads,cellmap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    NHF = number_of_basis_functions(sbasis)

    jac = face_determinant_jacobian(cellmap)
    nfaces = length(jac)
    @assert length(facequads) == nfaces

    ndofs = dim*NHF
    HH = [zeros(ndofs,ndofs) for i in 1:nfaces]

    for faceid in 1:nfaces
        if length(facequads[faceid]) > 0
            update_mass_matrix!(HH[faceid],sbasis,facequads[faceid],
                stabilization*jac[faceid],dim)
        end
    end

    return HH
end

function HHop!(HH,sbasis,squad,components::M,scale,
    nhdofs,NHF) where {M<:AbstractMatrix}

    NQ = length(squad)
    @assert size(components)[2] == NQ
    @assert size(HH) == (nhdofs*NHF,nhdofs*NHF)

    for (idx,(p,w)) in enumerate(squad)
        t = components[:,idx]
        projector = t*t'
        svals = sbasis(p)
        NP = make_row_matrix(svals,projector)
        NI = interpolation_matrix(svals,nhdofs)

        HH .+= scale*NP'*NI*w
    end

end

function HHop!(HH,sbasis,squad,components::V,scale,
    nhdofs,NHF) where {V<:AbstractVector}

    NQ = length(squad)
    extcomp = repeat(components,inner=(1,NQ))

    HHop!(HH,sbasis,squad,extcomp,scale,nhdofs,NHF)

end

function HHop_on_interface!(HH,sbasis,iquad,normals,imap,cellmap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    NHF = number_of_basis_functions(sbasis)

    NQ = length(iquad)
    @assert size(normals) == (dim,NQ)

    scale = scale_area(cellmap,normals)

    for (idx,(p,w)) in enumerate(iquad)
        vals = sbasis(p)
        N = interpolation_matrix(vals,dim)
        detjac = determinant_jacobian(imap,p)
        HH .+= N'*N*detjac*scale[idx]*w
    end
end

function HHop_on_interface!(HH,sbasis,iquad,components,normals,imap,cellmap,stabilization)

    sdim = dimension(sbasis)
    dim = sdim + 1
    NHF = number_of_basis_functions(sbasis)

    NQ = length(iquad)
    @assert size(normals) == (dim,NQ)
    @assert size(components) == (dim,NQ)

    scale = scale_area(cellmap,normals)

    for (idx,(p,w)) in enumerate(iquad)
        t = components[:,idx]
        projector = t*t'
        vals = sbasis(p)
        NP = make_row_matrix(vals,projector)
        NI = interpolation_matrix(vals,dim)
        detjac = determinant_jacobian(imap,p)
        HH .+= NP'*NI*detjac*scale[idx]*w
    end
    return HH
end
