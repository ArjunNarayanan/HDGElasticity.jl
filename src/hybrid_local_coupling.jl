function HLop!(HL,sbasis,vbasis,quad,linemap::LineMap,components::M,NED,
    scale,nhdofs) where {M<:AbstractMatrix}

    @assert size(components)[2] == length(quad)

    detjac = determinant_jacobian(linemap)

    for (idx,(p,w)) in enumerate(quad)
        t = components[:,idx]
        projector = t*t'
        svals = sbasis(p)
        vvals = vbasis(linemap(p...))
        NP = make_row_matrix(svals,projector)
        for nk in NED
            NK = make_row_matrix(vvals,nk)
            HL .+= NP'*NK*detjac*scale*w
        end
    end
end

function HLop!(HL,sbasis,vbasis,quad,linemap::LineMap,comp::V,NED,
    scale,nhdofs) where {V<:AbstractVector}

    nq = length(quad)
    extcomp = repeat(comp,inner=(1,nq))
    HLop!(HL,sbasis,vbasis,quad,linemap,extcomp,NED,scale,nhdofs)
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



function HUop!(HU,sbasis,rvbasis,squad,components::M,scale,
    detjac,nudofs,NHF,NUF) where {M<:AbstractMatrix}

    NQ = length(squad)
    @assert size(components)[2] == NQ
    @assert size(HU) == (nudofs*NHF,nudofs*NUF)

    for (idx,(p,w)) in enumerate(squad)
        t = components[:,idx]
        projector = t*t'
        svals = sbasis(p)
        vvals = rvbasis(p)
        NP = make_row_matrix(svals,projector)
        NI = interpolation_matrix(vvals,nudofs)

        HU .+= scale*NP'*NI*detjac*w
    end
end

# function HUop!(HU,sbasis,rvbasis,squad,components::V,scale,
#     detjac,nudofs,NHF,NUF) where {V<:AbstractVector}
#
#     NQ = length(squad)
#     extcomp = repeat(components,inner=(1,NQ))
#
#     HUop!(HU,sbasis,rvbasis,squad,extcomp,scale,detjac,nudofs,NHF,NUF)
#
# end
#
# function HUop(sbasis,vbasis,squad,components,faceid,cellmap,stabilization)
#
#     dim = dimension(vbasis)
#     NHF = number_of_basis_functions(sbasis)
#     NUF = number_of_basis_functions(vbasis)
#
#     cell = reference_cell(dim)
#     funcs = restrict_on_faces(vbasis,cell)
#     jac = jacobian(cellmap,cell)
#
#     HU = zeros(dim*NHF,dim*NUF)
#
#     HUop!(HU,sbasis,funcs[faceid],squad,components,stabilization,
#         jac[faceid],dim,NHF,NUF)
#
#     return HU
# end
#
# function hybrid_local_operator(sbasis,vbasis,squad,components,normals,
#     Dhalf,faceid,cellmap,stabilization)
#
#     HL = HLop(sbasis,vbasis,squad,components,normals,Dhalf,faceid,cellmap)
#     HU = HUop(sbasis,vbasis,squad,components,faceid,cellmap,stabilization)
#
#     return [HL HU]
#
# end
