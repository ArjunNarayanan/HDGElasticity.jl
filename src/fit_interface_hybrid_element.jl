function roots_of_restrictions(funcs,xL,xR)
    _roots = ImplicitDomainQuadrature.unique_roots.(funcs,xL,xR)
    lengths = length.(_roots)
    @assert all([l == 0 || l == 1 for l in lengths])
    roots = vcat(_roots...)
    faceids = vcat([repeat([fid],length(r)) for (fid,r) in enumerate(_roots)]...)
    return roots,faceids
end

function extend_to_face(x,faceid,cell::IntervalBox{2})
    if faceid == 1
        return extend(x,2,cell[2].lo)
    elseif faceid == 2
        return extend(x,1,cell[1].hi)
    elseif faceid == 3
        return extend(x,2,cell[2].hi)
    elseif faceid == 4
        return extend(x,1,cell[1].lo)
    else
        throw(ArgumentError("Expected faceid ∈ {1,2,3,4}, got faceid = $faceid"))
    end
end

function extend_face_roots(dim,roots,faceids,cell)
    extended_roots = hcat([extend_to_face(r,f,cell) for (r,f) in zip(roots,faceids)]...)
    return extended_roots
end

function element_face_intersections(poly::InterpolatingPolynomial{dim},
    coeffs) where {dim}

    xL,xR = reference_cell(dim)
    funcs = restrict_on_faces(poly,xL[1],xR[1])
    roots,indices = roots_of_restrictions(funcs,xL[1],xR[1])
end
