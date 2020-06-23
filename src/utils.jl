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

function reference_normals()
    return [[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]]
end

function reference_cell_volume(dim)
    return reference_element_length()^dim
end

function affine_map_jacobian(element_size)
    dim = length(element_size)
    reference_vol = reference_cell_volume(dim)
    return prod(element_size)/reference_vol
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

function nodal_coordinates(mesh::UniformMesh{dim,FT},
    basis::TensorProductBasis{dim,T,NF}) where {dim,T,NF,FT}

    ncells = mesh.total_number_of_elements
    coords = Matrix{FT}(undef,dim,NF*ncells)
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

    fb(x) = func(x,box[2].lo)
    fr(y) = func(box[1].hi,y)
    ft(x) = func(x,box[2].hi)
    fl(y) = func(box[1].lo,y)

    return [fb,fr,ft,fl]
end

function dimension(poly::InterpolatingPolynomial{1,NF,B}) where
    {NF,B<:TensorProductBasis{dim}} where {dim}

    return dim
end

function number_of_basis_functions(basis::TensorProductBasis{dim,T,NF}) where
    {dim,T,NF}

    return NF
end

function number_of_quadrature_points(quad::QuadratureRule{D,Q}) where {D,Q}
    return Q
end

function Base.length(quad::QuadratureRule)
    return number_of_quadrature_points(quad)
end
