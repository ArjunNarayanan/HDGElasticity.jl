function default_integer_type()
    return typeof(1)
end

function default_float_type()
    return typeof(1.0)
end

function reference_element_length()
    a,b = reference_interval_1d()
    return b-a
end

function in_reference_interval(x)
    a,b = reference_interval_1d()
    return a <= x <= b
end

function reference_cell(dim)
    if dim == 1
        xL = [-1.0]
        xR = [+1.0]
        return IntervalBox(xL,xR)
    elseif dim == 2
        xL = [-1.0,-1.0]
        xR = [+1.0,+1.0]
        return IntervalBox(xL,xR)
    else
        throw(ArgumentError("Expected dim ∈ {1,2}, got dim = $dim"))
    end
end

function number_of_faces(dim)
    if dim == 2
        return 4
    elseif dim == 3
        return 6
    else
        throw(ArgumentError("Expected dim ∈ {2,3}, got $dim == dim"))
    end
end

function reference_normals()
    return [[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]]
end

function reference_cell_volume(dim)
    return reference_element_length()^dim
end

function neighbor_faceid(faceid)
    if faceid == 1
        return 3
    elseif faceid == 2
        return 4
    elseif faceid == 3
        return 1
    elseif faceid == 4
        return 2
    else
        throw(ArgumentError("Expected faceid ∈ {1,2,3,4}, got faceid = $faceid"))
    end
end

function jacobian(map::InterpolatingPolynomial,p)
    return gradient(map,p)
end

function determinant_jacobian(map::InterpolatingPolynomial{2},p)
    return norm(jacobian(map,p))
end

function nodal_coordinates(mesh,basis)

    dim = dimension(mesh)
    @assert dimension(basis) == dim
    NF = number_of_basis_functions(basis)

    ncells = mesh.total_number_of_elements
    coords = zeros(dim,NF*ncells)
    start = 1
    stop = NF
    for idx in 1:ncells
        xL,xR = CartesianMesh.element(mesh,idx)
        map = CellMap(xL,xR)
        coords[:,start:stop] = map(basis.points)
        start = stop+1
        stop += NF
    end
    return coords
end

function restrict_on_faces(func,box::IntervalBox{2})

    fb(x) = func(extend(x,2,box[2].lo))
    fr(y) = func(extend(y,1,box[1].hi))
    ft(x) = func(extend(x,2,box[2].hi))
    fl(y) = func(extend(y,1,box[1].lo))

    return [fb,fr,ft,fl]
end

function dimension(basis::TensorProductBasis{dim}) where {dim}
    return dim
end

function dimension(poly::InterpolatingPolynomial)
    return dimension(poly.basis)
end

function dimension(quad::QuadratureRule{dim}) where dim
    return dim
end

function dimension(mesh::UniformMesh{dim}) where dim
    return dim
end

function number_of_basis_functions(basis::T) where
    {T<:PolynomialBasis.AbstractBasis{dim,NF}} where {dim,NF}

    return NF
end

function number_of_quadrature_points(quad::QuadratureRule{D,Q}) where {D,Q}
    return Q
end

function Base.length(quad::QuadratureRule)
    return number_of_quadrature_points(quad)
end

function levelset_normal(poly,p::V,cellmap) where {V<:AbstractVector}
    g = vec(gradient(poly,p))
    invjac = inverse_jacobian(cellmap)
    n = diagm(invjac)*g
    return n/norm(n)
end

function levelset_normal(poly,p::M,cellmap) where {M<:AbstractMatrix}
    g = hcat([gradient(poly,p[:,i])' for i in 1:size(p)[2]]...)
    invjac = inverse_jacobian(cellmap)
    normals = diagm(invjac)*g
    for i in 1:size(normals)[2]
        n = normals[:,i]
        normals[:,i] = n/norm(n)
    end
    return normals
end

function tangents(normal::V) where {V<:AbstractVector}
    @assert length(normal) == 2
    return [normal[2],-normal[1]]
end

function tangents(normals::M) where {M<:AbstractMatrix}
    m,n = size(normals)
    @assert m == 2

    t = similar(normals)
    t[1,:] = normals[2,:]
    t[2,:] = -normals[1,:]
    return t
end

function scale_area(cellmap,normals)
    t = tangents(normals)
    invjac = inverse_jacobian(cellmap)
    den = sqrt.((t.^2)'*(invjac.^2))
    return 1.0 ./ den
end

function active_faces(facequads)
    isactiveface = [length(fq) > 0 ? true : false for fq in facequads]
end

function face_midpoints(cellmap::CellMap{2})
    xl = cellmap.xL
    xr = cellmap.xR
    xm = 0.5*(xl+xr)

    m1 = [xm[1],xl[2]]
    m2 = [xr[1],xm[2]]
    m3 = [xm[1],xr[2]]
    m4 = [xl[1],xm[2]]

    return [m1,m2,m3,m4]
end

function levelset_coefficients(distance_function,mesh,basis)
    coords = nodal_coordinates(mesh,basis)
    NF = number_of_basis_functions(basis)
    return reshape(distance_function(coords),NF,:)
end
