function roots_of_restrictions(funcs,xL,xR)
    roots = ImplicitDomainQuadrature.unique_roots.(funcs,xL,xR)
    flags = [!isempty(r) for r in roots]
    @assert count(flags) == 2
    indices = findall(flags)
    return roots,indices
end

function extend_to_face(x,faceid,xL,xR)
    if faceid == 1
        return extend(x,2,xL)
    elseif faceid == 2
        return extend(x,1,xR)
    elseif faceid == 3
        return extend(x,2,xR)
    elseif faceid == 4
        return extend(x,1,xL)
    else
        throw(ArgumentError("Expected faceid ∈ {1,2,3,4}, got faceid = $faceid"))
    end
end

function extend_face_roots(dim,roots,indices,xL,xR)
    numroots = length(indices)
    extended_roots = zeros(dim,numroots)
    for (count,idx) in enumerate(indices)
        extended_roots[:,count] .= extend_to_face(roots[idx],idx,xL,xR)
    end
    return extended_roots
end

function element_face_intersections(poly::InterpolatingPolynomial{dim},
    coeffs) where {dim}

    xL,xR = reference_cell(dim)
    funcs = restrict_on_faces(poly,xL[1],xR[1])
    roots,indices = roots_of_restrictions(funcs,xL[1],xR[1])
end
