function roots_of_restrictions(funcs,xL,xR)
    roots = ImplicitDomainQuadrature.unique_roots.(funcs,xL,xR)
    flags = [!isempty(r) for r in roots]
    @assert count(flags) == 2
    indices = findall(flags)
    return roots,indices
end


function element_face_intersections(poly::InterpolatingPolynomial{dim},
    coeffs) where {dim}

    xL,xR = reference_cell(dim)
    funcs = restrict_on_faces(poly,xL[1],xR[1])

end
