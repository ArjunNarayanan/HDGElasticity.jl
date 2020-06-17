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
            quad = quadrature([funcs[faceid]],[sign_condition],xL,xR,quad1d)
            facequads[faceid,phase,cellidx] = quad
            visited[faceid,phase,cellidx] = true
        end
    end

end

function update_neighbor_face_quadrature!(facequads,visited,phase,cellid,
    nbrcellids)

    for (faceid,nbrcellid) in nbrcellids
        if nbrcellid != 0
            nbrfaceid = neighbor_faceid(faceid)
            if !visited[nbrfaceid,phase,nbrcellid]
                visited[nbrfaceid,phase,nbrcellid] = true
                facequads[nbrfaceid,phase,nbrcellid] = facequads[faceid,phase,cellid]
            end
        end
    end

end

function face_quadratures(dim,isactivecell,isactiveface,connectivity,
    coeffs,poly,quad1d)

    xL,xR = reference_cell(dim)
    box = IntervalBox(xL,xR)

    tpq = tensor_product(quad1d,box)

    facequads = similar(isactiveface, QuadratureRule{dim})

    visited = similar(isactiveface)
    fill!(visited,false)

    for idx in 1:ncells
        if isactivecell[1,idx] && !isactivecell[2,idx]
            facequads[:,1,idx] .= tpq
        elseif !isactivecell[2,idx] && isactivecell[2,idx]
            facequads[:,2,idx] .= tpq
        elseif isactivecell[1,idx] && isactivecell[2,idx]

            update!(poly,coeffs[:,idx])
            fb(x) = poly(x,xL[1])
            fr(y) = poly(xR[1],y)
            ft(x) = poly(x,xR[1])
            fl(y) = poly(xL[1],y)

            funcs = [fb,fr,ft,fl]

            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,+1,1,idx,xL[1],xR[1],quad1d)
            update_face_quadrature!(facequads,isactiveface,visited,
                funcs,-1,2,idx,xL[1],xR[1],quad1d)

            nbrcellids = [connectivity[i,idx] for i = 1:4]

            update_neighbor_face_quadrature!(facequads,visited,phase,
                cellid,nbrcellids)

        end
    end

end
