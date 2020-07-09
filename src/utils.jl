function default_integer_type()
    return typeof(1)
end

function default_float_type()
    return typeof(1.0)
end

function reference_element_length()
    return 2.0
end

function in_reference_interval(x)
    return -1.0 <= x <= +1.0
end

function reference_interval_1d()
    return (-1.0,+1.0)
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

struct AffineMap{dim,T}
    xL::SVector{dim,T}
    xR::SVector{dim,T}
    function AffineMap(xL::SVector{dim,T},xR::SVector{dim,T}) where {dim,T}
        @assert 1 <= dim <= 3
        @assert all(xL .< xR)
        new{dim,T}(xL,xR)
    end
end

function AffineMap(xL,xR)
    dim = length(xL)
    @assert length(xR) == dim
    sxL = SVector{dim}(xL)
    sxR = SVector{dim}(xR)
    return AffineMap(sxL,sxR)
end

function AffineMap(box::IntervalBox)
    xL = [int.lo for int in box.v]
    xR = [int.hi for int in box.v]
    return AffineMap(xL,xR)
end

function (M::AffineMap)(xi)
    return M.xL .+ 0.5*(1.0 .+ xi) .* (M.xR - M.xL)
end

function jacobian(M::AffineMap)
    return (M.xR - M.xL)/reference_element_length()
end

function jacobian(M::AffineMap{2},cell::IntervalBox{2})
    j = jacobian(M)
    return [j[1],j[2],j[1],j[2]]
end

function jacobian(map::InterpolatingPolynomial,p)
    return gradient(map,p)
end

function inverse_jacobian(M::AffineMap)
    return 1.0 ./ jacobian(M)
end

function determinant_jacobian(M::AffineMap)
    return prod(jacobian(M))
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
        map = AffineMap(xL,xR)
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
    return diagm(invjac)*g
end

function levelset_normal(poly,p::M,cellmap) where {M<:AbstractMatrix}
    g = hcat([gradient(poly,p[:,i])' for i in 1:size(p)[2]]...)
    invjac = inverse_jacobian(cellmap)
    return diagm(invjac)*g
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
