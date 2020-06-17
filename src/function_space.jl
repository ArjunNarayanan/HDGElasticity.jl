struct UniformFunctionSpace{vdim,sdim}
    vbasis::TensorProductBasis{vdim}
    sbasis::TensorProductBasis{sdim}
end


function element_quadratures!(equads,dim,isactivecell,coeffs,poly,quad1d)

    nf,ncells = size(coeffs)
    @assert size(isactivecell) == (2,ncells)

    xL,xR = reference_cell(dim)
    box = IntervalBox(xL,xR)

    tpq = tensor_product(quad1d,box)

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

function element_quadratures(dim,isactivecell,coeffs,poly,quad1d)
    equads = similar(isactivecell, QuadratureRule{dim})
    element_quadratures!(equads,dim,isactivecell,coeffs,poly,quad1d)
    return equads
end

function update_face_quadrature!(facequads,isactiveface,visited,
    funcs,sign_condition,phase,cellidx,xL,xR,quad1d)

    for (faceid,func) in enumerate(funcs)
        if isactiveface[faceid,phase,cellidx] && !visited[faceid,phase,cellidx]
            tempquad = quadrature([funcs[faceid]],[sign_condition],xL,xR,quad1d)
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

function face_quadratures!(facequads,dim,isactivecell,isactiveface,connectivity,
    coeffs,poly,quad1d)

    nphase,ncells = size(isactivecell)
    nfaces,_nphase,_ncells = size(isactiveface)
    @assert nphase == _nphase
    @assert ncells == _ncells

    xL,xR = reference_cell(dim)
    box = IntervalBox(xL,xR)

    tpq = tensor_product(quad1d,box)

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
            funcs = restrict_on_faces(poly,xL[1],xR[1])

            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,+1,1,cellid,xL[1],xR[1],quad1d)
            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,-1,2,cellid,xL[1],xR[1],quad1d)

            update_neighbor_face_quadrature!(facequads,isactiveface,visited,1,
                cellid,connectivity,nfaces)
            update_neighbor_face_quadrature!(facequads,isactiveface,visited,2,
                cellid,connectivity,nfaces)

        end
    end

end

function face_quadratures(dim,isactivecell,isactiveface,connectivity,coeffs,
    poly,quad1d)

    facequads = similar(isactiveface,QuadratureRule{dim})
    face_quadratures!(facequads,dim,isactivecell,isactiveface,connectivity,
        coeffs,poly,quad1d)
    return facequads

end
