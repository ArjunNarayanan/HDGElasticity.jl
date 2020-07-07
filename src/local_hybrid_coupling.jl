function update_LHop!(LH,rvbasis,sbasis,squad,NkEkD,detjac,nhdofs)

    for (p,w) in squad
        vals = rvbasis(p)
        svals = sbasis(p)

        Mk = make_row_matrix(vals,NkEkD)
        N = interpolation_matrix(svals,nhdofs)

        LH .+= Mk'*N*detjac*w
    end
end

function update_LHop!(LH,vbasis,sbasis,squad,nk,EkD,imap,nhdofs)

    for (idx,(p,w)) in enumerate(squad)
        vals = vbasis(imap(p))
        svals = sbasis(p)
        detjac = determinant_jacobian(imap,p)

        Mk = make_row_matrix(vals,EkD)
        N = interpolation_matrix(svals,nhdofs)

        LH .+= nk[idx]*Mk'*N*detjac*w
    end

end

function LHop!(LH,rvbasis,sbasis,squad,normal,Dhalf,Ek,detjac,nldofs,nhdofs,NLF,NHF)

    @assert size(Dhalf) == (nldofs,nldofs)
    sdim = dimension(squad)
    dim = sdim + 1
    @assert length(normal) == dim
    @assert length(Ek) == dim
    @assert size(LH) == (nldofs*NLF,nhdofs*NHF)

    for k = 1:dim
        NkEkD = normal[k]*Ek[k]'*Dhalf
        update_LHop!(LH,rvbasis,sbasis,squad,NkEkD,detjac,nhdofs)
    end
    return LH
end

function LHop_on_active_faces(vbasis,sbasis,facequads,isactiveface,Dhalf,cellmap)

    dim = dimension(vbasis)
    sdim = symmetric_tensor_dimension(dim)
    Ek = vec_to_symm_mat_converter(dim)
    NF = number_of_basis_functions(vbasis)
    NHF = number_of_basis_functions(sbasis)

    cell = reference_cell(dim)
    funcs = restrict_on_faces(vbasis,cell)
    jac = jacobian(cellmap,cell)

    normals = reference_normals()
    nfaces = length(funcs)

    @assert length(facequads) == nfaces
    @assert length(isactiveface) == nfaces

    LH = [zeros(sdim*NF,dim*NHF) for i = 1:nfaces]

    for faceid in 1:nfaces
        if isactiveface[faceid]
            LHop!(LH[faceid],funcs[faceid],sbasis,facequads[faceid],
                normals[faceid],Dhalf,Ek,jac[faceid],sdim,dim,NF,NHF)
        end
    end

    return LH
end

function LHop_on_interface(vbasis,sbasis,squad,normals,Dhalf,imap)

    dim = dimension(vbasis)
    sdim = symmetric_tensor_dimension(dim)
    nq = length(squad)

    @assert size(Dhalf) == (sdim,sdim)
    @assert size(normals) == (dim,nq)

    Ek = vec_to_symm_mat_converter(dim)
    NF = number_of_basis_functions(vbasis)
    NHF = number_of_basis_functions(sbasis)

    LH = zeros(sdim*NF,dim*NHF)

    for k = 1:dim
        EkD = Ek[k]'*Dhalf
        nk = normals[k,:]
        update_LHop!(LH,vbasis,sbasis,squad,nk,EkD,imap,dim)
    end

    return LH
end

function UHop!(UH,rvbasis,sbasis,squad,stabilization,detjac,nudofs,NF,NHF)

    for (p,w) in squad
        vals = rvbasis(p)
        svals = sbasis(p)

        Nv = interpolation_matrix(vals,nudofs)
        Ns = interpolation_matrix(svals,nudofs)

        UH .+= stabilization*Nv'*Ns*detjac*w
    end
end

function UHop_on_active_faces(vbasis,sbasis,facequads,isactiveface,
    stabilization,cellmap)

    dim = dimension(vbasis)
    sdim = dimension(sbasis)
    @assert sdim == dim-1

    NF = number_of_basis_functions(vbasis)
    NHF = number_of_basis_functions(sbasis)

    cell = reference_cell(dim)
    funcs = restrict_on_faces(vbasis,cell)
    jac = jacobian(cellmap,cell)

    nfaces = length(funcs)

    @assert length(facequads) == nfaces
    @assert length(isactiveface) == nfaces

    UH = [zeros(dim*NF,dim*NHF) for i = 1:nfaces]

    for faceid in 1:nfaces
        if isactiveface[faceid]
            UHop!(UH[faceid],funcs[faceid],sbasis,facequads[faceid],
                stabilization,jac[faceid],dim,NF,NHF)
        end
    end
    return UH
end
