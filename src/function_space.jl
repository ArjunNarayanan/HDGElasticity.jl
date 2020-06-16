struct FunctionSpace{dim}
    bases::Matrix{TensorProductBasis{dim}}
end

function element_bases(dim::Int,orders::M) where {M<:AbstractMatrix{Int}}
    nphase,ncells = size(orders)
    @assert nphase == 2
    @assert all(orders .> 0)

    unique_orders = unique(orders)
    tpbs = [TensorProductBasis(dim,o) for o in unique_orders]
    order2idx = Dict([unique_orders[i]=>i for i = 1:length(unique_orders)]...)
    bases = Matrix{TensorProductBasis{dim}}(undef,2,ncells)

    for idx = 1:ncells
        for phase = 1:2
            bases[phase,idx] = tpbs[order2idx[orders[phase,idx]]]
        end
    end
    return bases
end

function element_bases(dim::Int,order::Int,ncells::Int)
    orders = repeat([order],inner=(2,ncells))
    return element_bases(dim,orders)
end

function element_quadratures(dim,isactivecell,coeffs,poly,nqps::M) where
    {M<:AbstractMatrix}

    nf,ncells = size(coeffs)
    @assert size(isactivecell) == (2,ncells)
    @assert all(nqps .> 0)

    xL,xR = reference_cell(dim)
    box = IntervalBox(xL,xR)

    unique_nqp = unique(nqps)

    quad1d = [ImplicitDomainQuadrature.ReferenceQuadratureRule(nq) for nq in unique_nqp]
    tpqs = [tensor_product(q,box) for q in quad1d]

    nq2idx = Dict([unique_nqp[i]=>i for i = 1:length(unique_nqp)]...)

    quads = Matrix{QuadratureRule{dim}}(undef,2,ncells)


    for idx in 1:ncells
        if isactivecell[1,idx] && !isactivecell[2,idx]
            quads[1,idx] = tpqs[nq2idx[nqps[1,idx]]]
        elseif !isactivecell[1,idx] && isactivecell[2,idx]
            quads[2,idx] = tpqs[nq2idx[nqps[2,idx]]]
        elseif isactivecell[1,idx] && isactivecell[2,idx]

            update!(poly,coeffs[:,idx])

            nqplus = nqps[1,idx]
            qplus = quadrature(poly,+1,false,box,quad1d[nq2idx[nqplus]])
            quads[1,idx] = qplus

            nqminus = nqps[2,idx]
            qminus = quadrature(poly,-1,false,box,quad1d[nq2idx[nqminus]])
            quads[2,idx] = qminus

        end
    end
    return quads
end
