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

function extend_face_roots(roots,faceids,cell)
    extended_roots = hcat([extend_to_face(r,f,cell) for (r,f) in zip(roots,faceids)]...)
    return extended_roots
end

function roots_on_edges(funcs,xL,xR,yL,yR)
    r1 = find_zeros(funcs[1],xL,xR)
    f1 = repeat([1],length(r1))

    r2 = find_zeros(funcs[2],yL,yR)
    f2 = repeat([2],length(r2))

    r3 = find_zeros(funcs[3],xL,xR)
    f3 = repeat([3],length(r3))

    r4 = find_zeros(funcs[4],yL,yR)
    f4 = repeat([4],length(r4))

    r = vcat(r1,r2,r3,r4)
    f = vcat(f1,f2,f3,f4)

    return r,f
end

function element_face_intersections(poly,cell::IntervalBox{2})

    xL,xR = (cell[1].lo,cell[1].hi)
    yL,yR = (cell[2].lo,cell[2].hi)

    funcs = restrict_on_faces(poly,cell)

    roots,faceids = roots_on_edges(funcs,xL,xR,yL,yR)

    return extend_face_roots(roots,faceids,cell)
end
