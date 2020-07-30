struct UniformFunctionSpace{vdim,sdim,T}
    vbasis::TensorProductBasis{vdim}
    sbasis::TensorProductBasis{sdim}
    vtpq::QuadratureRule{vdim}
    ftpq::QuadratureRule{sdim}
    vquads::Matrix{QuadratureRule{vdim}}
    fquads::Array{QuadratureRule{sdim},3}
    icoeffs::Matrix{T}
    iquad::QuadratureRule{sdim}
    imap::InterpolatingPolynomial{vdim}
    inormals::Array{T,3}
    function UniformFunctionSpace(vbasis::TensorProductBasis{vdim},
        sbasis::TensorProductBasis{sdim},
        vtpq::QuadratureRule{vdim},ftpq::QuadratureRule{sdim},
        vquads::Matrix{QuadratureRule{vdim}},
        fquads::Array{QuadratureRule{sdim},3},icoeffs::Matrix{T},
        iquad::QuadratureRule{sdim},
        imap::InterpolatingPolynomial{vdim},
        inormals::Array{T,3}) where {vdim,sdim,NQ,T}

            @assert sdim == vdim-1
            ndofs,ncells = size(icoeffs)

            @assert size(vquads) == (2,ncells)
            nfaces = number_of_faces(vdim)
            @assert size(fquads) == (nfaces,2,ncells)

            nf = number_of_basis_functions(vbasis.basis)
            @assert number_of_basis_functions(sbasis.basis) == nf


            return new{vdim,sdim,T}(vbasis,sbasis,vtpq,ftpq,vquads,
                fquads,icoeffs,iquad,imap,inormals)

        end
end

function interface_normals(isactivecell,icoeffs,imap,lcoeffs,lpoly,
    refpoints,cellmap)

    sdim,nqp = size(refpoints)
    dim = sdim+1
    ncells = size(isactivecell)[2]
    inormals = zeros(dim,nqp,ncells)

    for cellid in 1:ncells
        if isactivecell[1,cellid] && isactivecell[2,cellid]
            update!(lpoly,lcoeffs[:,cellid])
            update!(imap,icoeffs[:,cellid])

            mappedpoints = hcat([imap(refpoints[:,i]) for i in 1:size(refpoints)[2]]...)
            inormals[:,:,cellid] = levelset_normal(lpoly,mappedpoints,cellmap)
        end
    end
    return inormals
end

function UniformFunctionSpace(dgmesh::DGMesh{vdim},polyorder,nquad,
    coeffs,poly) where {vdim}

    sdim = vdim-1

    cellmap = CellMap(dgmesh.domain[1])
    vbasis = TensorProductBasis(vdim,polyorder)
    sbasis = TensorProductBasis(sdim,polyorder)
    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(nquad)
    ftpq = tensor_product_quadrature(sdim,nquad)
    vtpq = tensor_product_quadrature(vdim,nquad)
    imap = InterpolatingPolynomial(vdim,sbasis)

    vquads = element_quadratures(dgmesh.isactivecell,coeffs,poly,quad1d)
    fquads = face_quadratures(dgmesh.isactivecell,dgmesh.isactiveface,
        dgmesh.connectivity,coeffs,poly,quad1d)
    icoeffs = interface_coefficients(dgmesh.isactivecell,coeffs,poly,sbasis,ftpq)
    inormals = interface_normals(dgmesh.isactivecell,icoeffs,imap,coeffs,poly,
        ftpq.points,cellmap)

    return UniformFunctionSpace(vbasis,sbasis,vtpq,ftpq,vquads,fquads,
        icoeffs,ftpq,imap,inormals)

end

function element_quadratures!(equads,isactivecell,coeffs,poly,tpq,quad1d)

    nf,ncells = size(coeffs)
    @assert size(isactivecell) == (2,ncells)

    dim = dimension(poly)
    box = reference_cell(dim)

    for idx in 1:ncells
        if isactivecell[1,idx] && !isactivecell[2,idx]
            equads[1,idx] = tpq
        elseif !isactivecell[1,idx] && isactivecell[2,idx]
            equads[2,idx] = tpq
        elseif isactivecell[1,idx] && isactivecell[2,idx]

            update!(poly,coeffs[:,idx])

            equads[1,idx] = quadrature(poly,+1,false,box,quad1d)
            equads[2,idx] = quadrature(poly,-1,false,box,quad1d)

        end
    end
end

function element_quadratures!(equads,isactivecell,coeffs,poly,quad1d)

    nf,ncells = size(coeffs)
    @assert size(isactivecell) == (2,ncells)

    dim = dimension(poly)
    box = reference_cell(dim)

    tpq = tensor_product(quad1d,box)

    element_quadratures!(equads,isactivecell,coeffs,poly,tpq,quad1d)
end

function element_quadratures(isactivecell,coeffs,poly,quad1d)
    dim = dimension(poly)
    equads = similar(isactivecell,QuadratureRule{dim})
    element_quadratures!(equads,isactivecell,coeffs,poly,quad1d)
    return equads
end

function update_face_quadrature!(facequads,isactiveface,visited,
    funcs,sign_condition,phase,cellidx,face::IntervalBox{1},quad1d)

    for (faceid,func) in enumerate(funcs)
        if isactiveface[faceid,phase,cellidx] && !visited[faceid,phase,cellidx]
            tempquad = quadrature([funcs[faceid]],[sign_condition],
                face[1].lo,face[1].hi,quad1d)
            quad = QuadratureRule(tempquad.points,tempquad.weights)
            facequads[faceid,phase,cellidx] = quad
            visited[faceid,phase,cellidx] = true
        end
    end

end

function update_neighbor_face_quadrature!(facequads,isactiveface,visited,
    phase,cellid,connectivity,nfaces)

    for faceid in 1:nfaces
        nbrcellid,nbrfaceid = connectivity[faceid,cellid]
        if nbrcellid != 0
            if isactiveface[nbrfaceid,phase,nbrcellid] && !visited[nbrfaceid,phase,nbrcellid]
                facequads[nbrfaceid,phase,nbrcellid] = facequads[faceid,phase,cellid]
                visited[nbrfaceid,phase,nbrcellid] = true
            end
        end
    end

end

function face_quadratures!(facequads,isactivecell,isactiveface,connectivity,
    coeffs,poly,quad1d)

    nphase,ncells = size(isactivecell)
    nfaces,_nphase,_ncells = size(isactiveface)
    @assert nphase == _nphase
    @assert ncells == _ncells

    dim = dimension(poly)
    cell = reference_cell(dim)
    face = reference_cell(dim-1)

    tpq = tensor_product(quad1d,face)

    visited = similar(isactiveface)
    fill!(visited,false)

    for cellid in 1:ncells
        if isactivecell[1,cellid] && !isactivecell[2,cellid]
            for faceid = 1:nfaces
                facequads[faceid,1,cellid] = tpq
            end
        elseif !isactivecell[2,cellid] && isactivecell[2,cellid]
            for faceid = 1:nfaces
                facequads[faceid,2,cellid] = tpq
            end
        elseif isactivecell[1,cellid] && isactivecell[2,cellid]

            update!(poly,coeffs[:,cellid])
            funcs = restrict_on_faces(poly,cell)

            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,+1,1,cellid,face,quad1d)
            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,-1,2,cellid,face,quad1d)

            update_neighbor_face_quadrature!(facequads,isactiveface,visited,1,
                cellid,connectivity,nfaces)
            update_neighbor_face_quadrature!(facequads,isactiveface,visited,2,
                cellid,connectivity,nfaces)

        end
    end

end

function face_quadratures(isactivecell,isactiveface,connectivity,coeffs,
    poly,quad1d)

    dim = dimension(poly)
    facedim = dim-1
    facequads = similar(isactiveface,QuadratureRule{facedim})
    face_quadratures!(facequads,isactivecell,isactiveface,connectivity,
        coeffs,poly,quad1d)
    return facequads

end

function interface_coefficients!(icoeffs,isactivecell,coeffs,poly,basis,quad)

    dim = dimension(poly)

    sdim = dimension(basis)
    @assert sdim == dim-1
    @assert dimension(quad) == sdim

    nphase,ncells = size(isactivecell)
    @assert nphase == 2
    nf = number_of_basis_functions(basis)
    @assert size(icoeffs) == (dim*nf,ncells)

    cell = reference_cell(dim)
    mass = lu(mass_matrix(basis,quad,dim,1.0))

    for cellid in 1:ncells
        if isactivecell[1,cellid] && isactivecell[2,cellid]
            update!(poly,coeffs[:,cellid])
            icoeffs[:,cellid] = fit_zero_levelset(poly,basis,quad,mass,cell)
        end
    end

end

function interface_coefficients(isactivecell,coeffs,poly,basis,quad)
    nf = number_of_basis_functions(basis)
    dim = dimension(poly)
    ncells = size(isactivecell)[2]
    icoeffs = zeros(dim*nf,ncells)
    interface_coefficients!(icoeffs,isactivecell,coeffs,poly,basis,quad)
    return icoeffs
end
