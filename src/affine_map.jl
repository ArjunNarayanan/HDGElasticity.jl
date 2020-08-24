function linear_map_slope(xiL,xiR,xL,xR)
    return (xR-xL)./(xiR-xiL)
end

function reference_interval_1d()
    return (-1.0,+1.0)
end

function reference_interval(dim)
    a,b = reference_interval_1d()
    xiL = a*ones(dim)
    xiR = b*ones(dim)
    return xiL,xiR
end

struct LineMap{dim,T}
    xiL::T
    xiR::T
    xL::MVector{dim,T}
    xR::MVector{dim,T}
    slope::MVector{dim,T}
    function LineMap(xiL::T,xiR::T,
        xL::MVector{dim,T},xR::MVector{dim,T}) where {dim,T}

        @assert 1 <= dim <= 3
        @assert xiL < xiR

        slope = linear_map_slope(xiL,xiR,xL,xR)

        new{dim,T}(xiL,xiR,xL,xR,slope)
    end
end

function LineMap(xiL,xiR,xL,xR)
    dim = length(xL)
    @assert length(xR) == dim
    sxL = MVector{dim}(xL)
    sxR = MVector{dim}(xR)
    return LineMap(xiL,xiR,sxL,sxR)
end

function dimension(M::LineMap{dim}) where {dim}
    return dim
end

function reference_interval(M::LineMap)
    return (M.xiL,M.xiR)
end

# function update!(M::LineMap,xL,xR)
#     dim = dimension(M)
#     @assert length(xL) == dim
#     @assert length(xR) == dim
#
#     M.xL .= xL
#     M.xR .= xR
#     M.slope .= linear_map_slope(M.xiL,M.xiR,M.xL,M.xR)
# end

function LineMap(xL,xR)
    xiL,xiR = reference_interval_1d()
    return LineMap(xiL,xiR,xL,xR)
end

function (M::LineMap)(xi)
    return M.xL .+ M.slope*(xi-M.xiL)
end

function jacobian(M::LineMap)
    return M.slope
end

function inverse_jacobian(M::LineMap)
    return 1.0 ./ jacobian(M)
end

function determinant_jacobian(M::LineMap)
    return norm(M.slope)
end

struct CellMap{dim,T}
    xiL::SVector{dim,T}
    xiR::SVector{dim,T}
    xL::MVector{dim,T}
    xR::MVector{dim,T}
    slope::MVector{dim,T}
    function CellMap(xiL::SVector{dim,T},xiR::SVector{dim,T},
        xL::MVector{dim,T},xR::MVector{dim,T}) where {dim,T}

        @assert 1 <= dim <= 3
        @assert all(xiL .< xiR)
        @assert all(xL .< xR)

        slope = linear_map_slope(xiL,xiR,xL,xR)

        new{dim,T}(xiL,xiR,xL,xR,slope)

    end
end

function CellMap(xiL,xiR,xL,xR)
    dim = length(xiL)
    @assert length(xiR) == dim
    @assert length(xL) == dim
    @assert length(xR) == dim

    sxiL = SVector{dim}(xiL)
    sxiR = SVector{dim}(xiR)
    mxL = MVector{dim}(xL)
    mxR = MVector{dim}(xR)
    return CellMap(sxiL,sxiR,mxL,mxR)
end

function CellMap(xi::IntervalBox{dim},x::IntervalBox{dim}) where {dim}
    xiL = [int.lo for int in xi]
    xiR = [int.hi for int in xi]
    xL = [int.lo for int in x]
    xR = [int.hi for int in x]
    CellMap(xiL,xiR,xL,xR)
end

function CellMap(x::IntervalBox{dim}) where {dim}
    xiL,xiR = reference_interval(dim)
    xL = [int.lo for int in x]
    xR = [int.hi for int in x]
    CellMap(xiL,xiR,xL,xR)
end

function CellMap(xL,xR)
    dim = length(xL)
    @assert length(xR) == dim
    xiL,xiR = reference_interval(dim)
    return CellMap(xiL,xiR,xL,xR)
end

function CellMap(dim::Z) where {Z<:Integer}
    xiL,xiR = reference_interval(dim)
    return CellMap(xiL,xiR,xiL,xiR)
end

function dimension(C::CellMap{dim}) where {dim}
    return dim
end

function update_range!(C::CellMap,xL,xR)
    dim = dimension(C)
    @assert length(xL) == dim
    @assert length(xR) == dim

    C.xL .= xL
    C.xR .= xR
    C.slope .= linear_map_slope(C.xiL,C.xiR,C.xL,C.xR)
end

function update_range!(C::CellMap{2},box::IntervalBox{2})
    C.xL[1] = box[1].lo
    C.xL[2] = box[2].lo
    C.xR[1] = box[1].hi
    C.xR[2] = box[2].hi

    C.slope .= linear_map_slope(C.xiL,C.xiR,C.xL,C.xR)
end

function (C::CellMap)(xi)
    return C.xL .+ C.slope .* (xi .- C.xiL)
end

function jacobian(C::CellMap)
    return C.slope
end

function face_determinant_jacobian(C::CellMap{2})
    j = jacobian(C)
    return [j[1],j[2],j[1],j[2]]
end

function inverse_jacobian(C::CellMap)
    return 1.0 ./ jacobian(C)
end

function determinant_jacobian(C::CellMap)
    return prod(jacobian(C))
end

function reference_cell_facemaps(dim)
    @assert dim == 2
    xL,xR = reference_interval(dim)
    m1 = LineMap([xL[1],xL[2]],[xR[1],xL[2]])
    m2 = LineMap([xR[1],xL[2]],[xR[1],xR[2]])
    m3 = LineMap([xL[1],xR[2]],[xR[1],xR[2]])
    m4 = LineMap([xL[1],xL[2]],[xL[1],xR[2]])
    return [m1,m2,m3,m4]
end
