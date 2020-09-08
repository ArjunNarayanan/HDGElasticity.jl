function HLop!(HL,sbasis,vbasis,quad,linemap::LineMap,components,NED,
    scale,nhdofs)

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
    components,normals,ED,scale,nhdofs)

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

function HUop!(HU,sbasis,vbasis,quad,linemap::LineMap,components,scale,
    nhdofs)

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

    @assert length(scale) == length(quad)
    @assert size(components)[2] == length(quad)

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
