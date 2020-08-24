struct UniformFunctionSpace{vdim,sdim}
    vbasis::TensorProductBasis{vdim}
    sbasis::TensorProductBasis{sdim}
    vtpq::QuadratureRule{vdim}
    ftpq::QuadratureRule{sdim}
    vquads
    fquads
    facemaps
    icoeffs
    iquad
    imap
    inormals
    function UniformFunctionSpace(vbasis::TensorProductBasis{vdim},
        sbasis::TensorProductBasis{sdim},vtpq,ftpq,vquads,fquads,facemaps,
        icoeffs,iquad,imap,inormals) where {vdim,sdim}

            @assert sdim == vdim-1
            nphase,ncells = size(vquads)

            @assert length(icoeffs) == ncells
            @assert length(facemaps) == ncells
            @assert size(fquads) == (nphase,ncells)
            nfaces = number_of_faces(vdim)
            @assert all(length.(fquads) .== nfaces)
            @assert all(length.(facemaps) .== nfaces)

            return new{vdim,sdim}(vbasis,sbasis,vtpq,ftpq,vquads,
                fquads,facemaps,icoeffs,iquad,imap,inormals)

        end
end

function UniformFunctionSpace(dgmesh,polyorder,quadorder,coeffs,levelset,
    facemaps)

    vdim = dimension(dgmesh)
    sdim = vdim-1

    vbasis = TensorProductBasis(vdim,polyorder)
    sbasis = TensorProductBasis(sdim,polyorder)

    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(quadorder)
    vtpq = tensor_product(quad1d,reference_cell(vdim))
    ftpq = tensor_product(quad1d,reference_cell(sdim))

    iquad = ftpq
    imap = InterpolatingPolynomial(vdim,sbasis)

    vquads = element_quadratures(dgmesh.cellsign,coeffs,levelset,quad1d)
    fquads = face_quadratures(dgmesh.cellsign,coeffs,levelset,facemaps,quad1d)

    icoeffs = interface_coefficients(dgmesh.cellsign,coeffs,levelset,
        sbasis,iquad)
    inormals = interface_normals(dgmesh,icoeffs,imap,coeffs,levelset,
        iquad.points)

    return UniformFunctionSpace(vbasis,sbasis,vtpq,ftpq,vquads,fquads,facemaps,
        icoeffs,iquad,imap,inormals)
end

function interface_normals(dgmesh,icoeffs,imap,lcoeffs,levelset,
    refpoints)

    sdim,nqp = size(refpoints)
    dim = sdim+1

    cellsign = dgmesh.cellsign

    ncells = length(cellsign)
    ft = default_float_type()
    inormals = Vector{Matrix{ft}}(undef,ncells)
    cellmap = CellMap(dim)

    for cellid in 1:ncells
        if cellsign[cellid] == 0
            update!(levelset,lcoeffs[:,cellid])
            update!(imap,icoeffs[cellid])
            update_range!(cellmap,dgmesh.domain[cellid])

            mappedpoints = hcat([imap(refpoints[:,i]) for i in 1:size(refpoints)[2]]...)
            inormals[cellid] = levelset_normal(levelset,mappedpoints,cellmap)
        end
    end
    return inormals
end

function element_quadratures!(equads,cellsign,coeffs,poly,tpq,box,quad1d)

    nf,ncells = size(coeffs)
    @assert length(cellsign) == ncells
    @assert size(equads) == (2,ncells)

    for idx in 1:ncells
        s = cellsign[idx]
        if s == +1
            equads[1,idx] = tpq
        elseif s == -1
            equads[2,idx] = tpq
        elseif s == 0
            update!(poly,coeffs[:,idx])
            equads[1,idx] = quadrature(poly,+1,false,box,quad1d)
            equads[2,idx] = quadrature(poly,-1,false,box,quad1d)
        else
            throw(ArgumentError("Expected `cellsign[idx] ∈ {-1,0,+1}`,
                got `cellsign[$idx] == $s`"))
        end
    end
end

function element_quadratures!(equads,cellsign,coeffs,poly,quad1d)

    nf,ncells = size(coeffs)
    @assert length(cellsign) == ncells
    @assert size(equads) == (2,ncells)

    dim = dimension(poly)
    box = reference_cell(dim)

    tpq = tensor_product(quad1d,box)

    element_quadratures!(equads,cellsign,coeffs,poly,tpq,box,quad1d)
end

function element_quadratures(cellsign,coeffs,poly,quad1d)
    dim = dimension(poly)
    ncells = length(cellsign)
    equads = Array{QuadratureRule{dim}}(undef,2,ncells)
    element_quadratures!(equads,cellsign,coeffs,poly,quad1d)
    return equads
end

function update_face_quadratures!(facequads,levelset,facemaps,
    sign_condition,quad1d)

    nfaces = length(facequads)
    @assert length(facemaps) == nfaces
    for faceid in 1:nfaces
        fmap = facemaps[faceid]
        xiL,xiR = reference_interval(fmap)
        quad = QuadratureRule(quadrature([x->levelset(fmap(x))],
            [sign_condition],xiL,xiR,quad1d))
        facequads[faceid] = quad
    end
end

function assign_all(array,value,ncells)
    for i = 1:ncells
        array[i] = value
    end
end

function face_quadratures!(facequads,cellsign,coeffs,levelset,facemaps,
    quad1d)

    nf,ncells = size(coeffs)
    @assert length(facemaps) == ncells
    @assert size(facequads) == (2,ncells)

    nfaces = [length(fm) for fm in facemaps]

    @assert all([length(facequads[1,i]) == nfaces[i] for i = 1:ncells])
    @assert all([length(facequads[2,i]) == nfaces[i] for i = 1:ncells])

    dim = dimension(levelset)
    reference_face = reference_cell(dim-1)

    tpq = tensor_product(quad1d,reference_face)

    for cellid in 1:ncells
        if cellsign[cellid] == +1
            assign_all(facequads[1,cellid],tpq,nfaces[cellid])
        elseif cellsign[cellid] == -1
            assign_all(facequads[2,cellid],tpq,nfaces[cellid])
        elseif cellsign[cellid] == 0

            update!(levelset,coeffs[:,cellid])

            update_face_quadratures!(facequads[1,cellid],levelset,
                facemaps[cellid],+1,quad1d)
            update_face_quadratures!(facequads[2,cellid],levelset,
                facemaps[cellid],-1,quad1d)

        end
    end

end

function face_quadratures(cellsign,coeffs,levelset,facemaps,quad1d)

    nfuncs,ncells = size(coeffs)
    @assert length(cellsign) == ncells
    @assert length(facemaps) == ncells

    nfaces = repeat([length(fm) for fm in facemaps],inner=(2,))

    dim = dimension(levelset)
    facedim = dim-1
    facequads = reshape([Vector{QuadratureRule{facedim}}(undef,nf) for nf in nfaces],2,:)
    face_quadratures!(facequads,cellsign,coeffs,levelset,facemaps,quad1d)
    return facequads

end

function interface_coefficients!(icoeffs,cellsign,coeffs,levelset,basis,quad)

    dim = dimension(levelset)

    sdim = dimension(basis)
    @assert sdim == dim-1
    @assert dimension(quad) == sdim

    ncells = length(cellsign)
    nf = number_of_basis_functions(basis)

    cell = reference_cell(dim)
    mass = lu(mass_matrix(basis,quad,1.0,dim))

    for cellid in 1:ncells
        if cellsign[cellid] == 0
            update!(levelset,coeffs[:,cellid])
            icoeffs[cellid] = fit_zero_levelset(levelset,basis,quad,mass,cell)
        end
    end

end

function interface_coefficients(cellsign,coeffs,levelset,basis,quad)
    nf = number_of_basis_functions(basis)
    dim = dimension(levelset)
    ncells = length(cellsign)
    ft = default_float_type()
    icoeffs = Vector{Vector{ft}}(undef,ncells)
    interface_coefficients!(icoeffs,cellsign,coeffs,levelset,basis,quad)
    return icoeffs
end
