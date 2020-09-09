function LHop!(LH,vbasis,sbasis,quad,linemap::LineMap,NED,scale,nhdofs)

    detjac = determinant_jacobian(linemap)
    for nk in NED
        for (p,w) in quad
            vals = vbasis(linemap(p...))
            svals = sbasis(p)

            M = make_row_matrix(vals,nk)
            N = interpolation_matrix(svals,nhdofs)

            LH .+= M'*N*detjac*scale*w
        end
    end
end

function LHop!(LH,vbasis,sbasis,quad,imap::InterpolatingPolynomial,normals,
    ED,scale,nhdofs)

    @assert length(scale) == length(quad)
    @assert size(normals)[2] == length(quad)

    for (idx,(p,w)) in enumerate(quad)
        n = view(normals,:,idx)
        vals = vbasis(imap(p))
        svals = sbasis(p)
        detjac = determinant_jacobian(imap,p)

        for k in 1:length(ED)
            NED = n[k]*ED[k]

            M = make_row_matrix(vals,NED)
            N = interpolation_matrix(svals,nhdofs)

            LH .+= M'*N*detjac*scale[idx]*w

        end
    end
end

function LHop(vbasis,sbasis,facequads,facemaps,normals,Dhalf,cellmap)

    nfaces = length(facemaps)
    @assert length(normals) == nfaces
    @assert length(facequads) == nfaces

    dim = dimension(vbasis)
    sdim = symmetric_tensor_dimension(dim)
    Ek = vec_to_symm_mat_converter(dim)

    NLF = number_of_basis_functions(vbasis)
    NHF = number_of_basis_functions(sbasis)

    isactiveface = active_faces(facequads)

    LH = [zeros(sdim*NLF,dim*NHF) for i in 1:count(isactiveface)]
    facescale = face_determinant_jacobian(cellmap)

    counter = 1
    for faceid in 1:nfaces
        if isactiveface[faceid]
            n = normals[faceid]
            NED = [n[k]*Ek[k]'*Dhalf for k in 1:length(Ek)]
            LHop!(LH[counter],vbasis,sbasis,facequads[faceid],facemaps[faceid],
                NED,facescale[faceid],dim)
            counter += 1
        end
    end
    return LH
end

function LHop_on_interface(vbasis,sbasis,iquad,imap,normals,Dhalf,cellmap)

    dim = dimension(vbasis)
    sdim = symmetric_tensor_dimension(dim)
    nq = length(iquad)

    @assert size(Dhalf) == (sdim,sdim)
    @assert size(normals) == (dim,nq)

    Ek = vec_to_symm_mat_converter(dim)
    NLF = number_of_basis_functions(vbasis)
    NHF = number_of_basis_functions(sbasis)

    LH = zeros(sdim*NLF,dim*NHF)
    scale = scale_area(cellmap,normals)

    ED = [E'*Dhalf for E in Ek]
    LHop!(LH,vbasis,sbasis,iquad,imap,normals,ED,scale,dim)

    return LH
end

function UHop!(UH,vbasis,sbasis,quad,linemap::LineMap,
    scale,nudofs)

    detjac = determinant_jacobian(linemap)
    for (p,w) in quad
        vals = vbasis(linemap(p...))
        svals = sbasis(p)

        Nv = interpolation_matrix(vals,nudofs)
        Ns = interpolation_matrix(svals,nudofs)

        UH .+= Nv'*Ns*detjac*scale*w
    end
end

function UHop!(UH,vbasis,sbasis,quad,imap::InterpolatingPolynomial,
    scale,nudofs)

    @assert length(scale) == length(quad)

    for (idx,(p,w)) in enumerate(quad)
        vals = vbasis(imap(p))
        svals = sbasis(p)
        detjac = determinant_jacobian(imap,p)

        Nv = interpolation_matrix(vals,nudofs)
        Ns = interpolation_matrix(svals,nudofs)

        UH .+= Nv'*Ns*detjac*scale[idx]*w
    end
end

function UHop(vbasis,sbasis,facequads,facemaps,stabilization,cellmap)

    nfaces = length(facemaps)
    @assert length(facequads) == nfaces

    dim = dimension(vbasis)
    sdim = dimension(sbasis)
    @assert sdim == dim-1

    NUF = number_of_basis_functions(vbasis)
    NHF = number_of_basis_functions(sbasis)

    isactiveface = [length(fq) > 0 ? true : false for fq in facequads]

    facejac = face_determinant_jacobian(cellmap)
    UH = [zeros(dim*NUF,dim*NHF) for i = 1:count(isactiveface)]

    counter = 1
    for faceid in 1:nfaces
        if isactiveface[faceid]
            UHop!(UH[counter],vbasis,sbasis,facequads[faceid],facemaps[faceid],
                stabilization*facejac[faceid],dim)
            counter += 1
        end
    end
    return UH
end

function UHop_on_interface(vbasis,sbasis,squad,imap,normals,
    stabilization,cellmap)

    dim = dimension(vbasis)
    sdim = dimension(sbasis)
    @assert sdim == dim-1

    NUF = number_of_basis_functions(vbasis)
    NHF = number_of_basis_functions(sbasis)

    UH = zeros(dim*NUF,dim*NHF)
    scale = stabilization*scale_area(cellmap,normals)

    UHop!(UH,vbasis,sbasis,squad,imap,scale,dim)

    return UH
end

function local_hybrid_operator(vbasis,sbasis,facequads,facemaps,normals,
    Dhalf,stabilization,cellmap)

    LH = LHop(vbasis,sbasis,facequads,facemaps,normals,Dhalf,cellmap)
    UH = UHop(vbasis,sbasis,facequads,facemaps,stabilization,cellmap)

    @assert length(LH) == length(UH)

    return hcat([[LH[i];UH[i]] for i in 1:length(LH)]...)

end

function local_hybrid_operator_on_interface(vbasis,sbasis,iquad,imap,inormals,
    Dhalf,stabilization,cellmap)

    LH = LHop_on_interface(vbasis,sbasis,iquad,imap,inormals,Dhalf,cellmap)
    UH = UHop_on_interface(vbasis,sbasis,iquad,imap,inormals,
        stabilization,cellmap)

    return [LH;UH]
end
