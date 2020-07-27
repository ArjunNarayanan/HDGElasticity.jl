function HLop!(HL,sbasis,rvbasis,squad,components::M1,normals::M2,Dhalf,Ek,
    detjac,nhdofs,nldofs,NHF,NLF) where {M1<:AbstractMatrix,M2<:AbstractMatrix}

    @assert size(Dhalf) == (nldofs,nldofs)
    NQ = length(squad)

    @assert size(normals)[2] == NQ
    @assert size(components)[2] == NQ
    @assert length(Ek) == size(normals)[1]
    @assert size(HL) == (nhdofs*NHF,nldofs*NLF)


    for (idx,(p,w)) in enumerate(squad)
        t = components[:,idx]
        projector = t*t'
        svals = sbasis(p)
        vvals = rvbasis(p)
        NP = make_row_matrix(svals,projector)
        for k = 1:length(Ek)
            NkEkD = normals[k,idx]*Ek[k]'*Dhalf
            NK = make_row_matrix(vvals,NkEkD)

            HL .+= NP'*NK*detjac*w
        end
    end

end

function HLop!(HL,sbasis,rvbasis,squad,comp::V1,normals::V2,Dhalf,Ek,
    detjac,nhdofs,nldofs,NHF,NLF) where {V1<:AbstractVector,V2<:AbstractVector}

    nq = length(squad)
    extcomp = repeat(comp,inner=(1,nq))
    extnormals = repeat(normals,inner=(1,nq))

    HLop!(HL,sbasis,rvbasis,squad,extcomp,extnormals,Dhalf,Ek,detjac,nhdofs,
        nldofs,NHF,NLF)
end

function HLop(sbasis,vbasis,squad,components,normals,Dhalf,faceid,cellmap)

    dim = dimension(vbasis)
    sdim = symmetric_tensor_dimension(dim)
    Ek = vec_to_symm_mat_converter(dim)
    NHF = number_of_basis_functions(sbasis)
    NF = number_of_basis_functions(vbasis)

    cell = reference_cell(dim)
    funcs = restrict_on_faces(vbasis,cell)
    jac = jacobian(cellmap,cell)

    HL = zeros(dim*NHF,sdim*NF)

    HLop!(HL,sbasis,funcs[faceid],squad,components,normals,Dhalf,Ek,
        jac[faceid],dim,sdim,NHF,NF)

    return HL

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

function HUop!(HU,sbasis,rvbasis,squad,components::V,scale,
    detjac,nudofs,NHF,NUF) where {V<:AbstractVector}

    NQ = length(squad)
    extcomp = repeat(components,inner=(1,NQ))

    HUop!(HU,sbasis,rvbasis,squad,extcomp,scale,detjac,nudofs,NHF,NUF)

end

function HUop(sbasis,vbasis,squad,components,faceid,cellmap,stabilization)

    dim = dimension(vbasis)
    NHF = number_of_basis_functions(sbasis)
    NUF = number_of_basis_functions(vbasis)

    cell = reference_cell(dim)
    funcs = restrict_on_faces(vbasis,cell)
    jac = jacobian(cellmap,cell)

    HU = zeros(dim*NHF,dim*NUF)

    HUop!(HU,sbasis,funcs[faceid],squad,components,stabilization,
        jac[faceid],dim,NHF,NUF)

    return HU
end
