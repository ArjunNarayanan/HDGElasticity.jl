struct FunctionSpace{dim}
    elid2basis::Vector{TensorProductBasis{dim}}
end

function bases(tpb::TensorProductBasis,nelements::Int)
    @assert nelements > 0
    return [tpb for i = 1:nelements]
end

function bases(dim::Int,order::Int,nelements::Int)
    tpb = TensorProductBasis(dim,order)
    return bases(tpb,nelements)
end

function bases(dim::Int,orders::V) where {V<:AbstractVector}
    unique_orders = unique(orders)
    tpbs = [TensorProductBasis(dim,o) for o in unique_orders]
    order2idx = Dict([unique_orders[i]=>i for i = 1:length(unique_orders)]...)
    tpbases = [tpbs[order2idx[o]] for o in orders]
    return tpbases
end
