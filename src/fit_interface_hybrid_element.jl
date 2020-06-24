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

function roots_without_end(f,xL,xR,atol)
    r = sort!(find_zeros(f,xL,xR))
    if length(r) > 0
        isapprox(r[end],xR,atol=atol) ? pop!(r) : nothing
    end
    return r
end

function roots_without_start(f,xL,xR,atol)
    r = sort!(find_zeros(f,xL,xR))
    if length(r) > 0
        isapprox(r[1],xL,atol=atol) ? popfirst!(r) : nothing
    end
    return r
end

function roots_on_edges(funcs,xL,xR,yL,yR;atol=0.0)
    r1 = roots_without_end(funcs[1],xL,xR,atol)
    f1 = repeat([1],length(r1))

    r2 = roots_without_end(funcs[2],yL,yR,atol)
    f2 = repeat([2],length(r2))

    r3 = roots_without_start(funcs[3],xL,xR,atol)
    f3 = repeat([3],length(r3))

    r4 = roots_without_start(funcs[4],yL,yR,atol)
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

    @assert length(roots) == 2

    x1 = extend_to_face(roots[1],faceids[1],cell)
    x2 = extend_to_face(roots[2],faceids[2],cell)

    return x1,x2
end


function gradient_descent_to_zero!(P,∇P,x,atol,maxiter)
    @assert atol > 0.0
    @assert maxiter > 0

    val = P(x)
    iter = 1
    while abs(val) > atol && iter < maxiter
        dval = vec(∇P(x))
        alpha = val/dot(dval,dval)
        x .-= alpha*dval
        val = P(x)
        iter += 1
    end
    if iter == maxiter
        throw(ErrorException("Newton iteration failed to converge in $maxiter iterations"))
    end
end

function gradient_descent_to_zero!(P::InterpolatingPolynomial,x,atol,maxiter)
    gradient_descent_to_zero!(P,x->gradient(P,x),x,atol,maxiter)
end

function resolve_zero_levelset(poly,refpoints,xL,xR;atol=1e-12,maxiter=50)
    dim = dimension(poly)
    @assert dim == 2
    np = length(refpoints)

    points = zeros(2,np)
    dx = xR - xL

    for (idx,p) in enumerate(refpoints)
        x = xL + 0.5*(1.0+p)*dx
        gradient_descent_to_zero!(poly,x,atol,maxiter)
        points[:,idx] = x
    end
    return points
end

function fit_zero_levelset(poly,basis1d,quad1d,mass,cell;atol=1e-12,maxiter=50)
    xL,xR = element_face_intersections(poly,cell)
    intp = resolve_zero_levelset(poly,quad1d.points,xL,xR,atol=atol,maxiter=maxiter)
    rhs = linear_form(intp,basis1d,quad1d)
    return mass\rhs
end
