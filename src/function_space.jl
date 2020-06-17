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
    phase,cellid,nbrcellids)

    for (faceid,nbrcellid) in enumerate(nbrcellids)
        if nbrcellid != 0
            nbrfaceid = neighbor_faceid(faceid)
            if isactiveface[faceid,phase,cellid] && !visited[nbrfaceid,phase,nbrcellid]
                visited[nbrfaceid,phase,nbrcellid] = true
                println(faceid," ",phase," ",cellid)
                facequads[nbrfaceid,phase,nbrcellid] = facequads[faceid,phase,cellid]
            end
        end
    end

end

function face_quadratures!(facequads,dim,isactivecell,isactiveface,connectivity,
    coeffs,poly,quad1d)

    nphase,ncells = size(isactivecell)
    nface,_nphase,_ncells = size(isactiveface)

    xL,xR = reference_cell(dim)
    box = IntervalBox(xL,xR)

    tpq = tensor_product(quad1d,box)

    visited = similar(isactiveface)
    fill!(visited,false)

    for cellid in 1:ncells
        if isactivecell[1,cellid] && !isactivecell[2,cellid]
            for faceid = 1:nface
                facequads[faceid,1,cellid] = tpq
            end
        elseif !isactivecell[2,cellid] && isactivecell[2,cellid]
            for faceid = 1:nface
                facequads[faceid,2,cellid] = tpq
            end
        elseif isactivecell[1,cellid] && isactivecell[2,cellid]

            update!(poly,coeffs[:,cellid])

            funcs = restrict_on_faces(poly,xL[1],xR[1])

            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,+1,1,cellid,xL[1],xR[1],quad1d)
            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,-1,2,cellid,xL[1],xR[1],quad1d)

            nbrcellids = [connectivity[faceid,cellid] for faceid = 1:nface]

            println(nbrcellids)

            update_neighbor_face_quadrature!(facequads,isactiveface,visited,1,
                cellid,nbrcellids)
            update_neighbor_face_quadrature!(facequads,isactiveface,visited,2,
                cellid,nbrcellids)

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
