function HLop!(HL,sbasis,vbasis,quad,linemap::LineMap,normals,components,
    ED,scale)

    @assert length(normals) == length(ED)
    @assert length(components) == length(ED)
    NED = [normals[k]*ED[k] for k = 1:length(normals)]

    detjac = determinant_jacobian(linemap)
    projector = components*components'

    for (idx,(p,w)) in enumerate(quad)
        svals = sbasis(p)
        vvals = vbasis(linemap(p...))
        NP = make_row_matrix(svals,projector)
        for nk in NED
            NK = make_row_matrix(vvals,nk)
            HL .+= NP'*NK*detjac*scale*w
        end
    end
end

function HLop!(HL,sbasis,vbasis,quad,imap::InterpolatingPolynomial,
    normals,components,ED,scale)

    NQ = length(quad)
    @assert length(scale) == NQ
    @assert size(normals)[2] == NQ
    @assert size(components)[2] == NQ
    @assert size(normals)[1] == length(ED)

    for (idx,(p,w)) in enumerate(quad)
        n = normals[:,idx]
        t = components[:,idx]
        projector = t*t'
        svals = sbasis(p)
        vvals = vbasis(imap(p))
        detjac = determinant_jacobian(imap,p)
        NP = make_row_matrix(svals,projector)
        for k in 1:length(ED)
            NED = n[k]*ED[k]
            NK = make_row_matrix(vvals,NED)

            HL .+= NP'*NK*detjac*scale[idx]*w
        end
    end
end

function HLop(sbasis,vbasis,facequad,facemap,normals,components,
    Dhalf,facescale)

    dim = dimension(vbasis)

    NHF = number_of_basis_functions(sbasis)
    NLF = number_of_basis_functions(vbasis)

    sdim = symmetric_tensor_dimension(dim)
    Ek = vec_to_symm_mat_converter(dim)
    ED = [Ek[k]'*Dhalf for k = 1:dim]

    HL = zeros(dim*NHF,sdim*NLF)

    HLop!(HL,sbasis,vbasis,facequad,facemap,normals,components,ED,facescale)

    return HL
end

function HUop!(HU,sbasis,vbasis,quad,linemap::LineMap,components,scale,
    nhdofs)

    @assert length(components) == nhdofs
    detjac = determinant_jacobian(linemap)
    projector = components*components'

    for (idx,(p,w)) in enumerate(quad)
        svals = sbasis(p)
        vvals = vbasis(linemap(p...))
        NP = make_row_matrix(svals,projector)
        NI = interpolation_matrix(vvals,nhdofs)

        HU .+= scale*NP'*NI*detjac*w
    end
end

function HUop!(HU,sbasis,vbasis,quad,imap::InterpolatingPolynomial,
    components,scale,nhdofs)

    NQ = length(quad)
    @assert length(scale) == NQ
    @assert size(components) == (nhdofs,NQ)

    for (idx,(p,w)) in enumerate(quad)
        t = components[:,idx]
        projector = t*t'
        svals = sbasis(p)
        vvals = vbasis(imap(p))
        detjac =  determinant_jacobian(imap,p)
        NP = make_row_matrix(svals,projector)
        NI = interpolation_matrix(vvals,nhdofs)

        HU .+= scale[idx]*NP'*NI*detjac*w
    end

end

function HUop(sbasis,vbasis,facequad,facemap,components,scale,nhdofs)

    dim = dimension(vbasis)

    NHF = number_of_basis_functions(sbasis)
    NLF = number_of_basis_functions(vbasis)

    HU = zeros(dim*NHF,dim*NLF)

    HUop!(HU,sbasis,vbasis,facequad,facemap,components,scale,nhdofs)

    return HU

end
