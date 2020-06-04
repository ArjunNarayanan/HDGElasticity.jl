struct DGMesh{dim,T}
    domain::Vector{IntervalBox{dim,T}}
    connectivity::Matrix{Int}
    isactivecell::Matrix{Bool}
    isactiveface::Matrix{Vector{Bool}}
    function DGMesh(domain::Vector{IntervalBox{dim,T}},connectivity::Matrix{T},
        isactivecell::Matrix{Bool},isactiveface::Matrix{Vector{Bool}}) where
        {dim,T}

        @assert dim == 2
        ncells = length(domain)
        @assert size(connectivity) == (4,ncells)
        @assert size(isactivecell) == (2,ncells)
        @assert size(isactiveface) == (2,ncells)
        @assert all(length.(isactiveface) .== 4)

        new{dim,T}(domain,connectivity,isactivecell,isactiveface)

    end
end

function cell_domain(mesh::UniformMesh{dim,T}) where {dim,T}
    domain = Vector{IntervalBox{dim,T}}(undef,mesh.total_number_of_elements)
    for idx in 1:mesh.total_number_of_elements
        xL,xR = CartesianMesh.element(mesh,idx)
        domain[idx] = IntervalBox(xL,xR)
    end
    return domain
end

function cell_connectivity(mesh)
    connectivity = Matrix{Int}(undef,4,mesh.total_number_of_elements)
    for idx in 1:mesh.total_number_of_elements
        connectivity[:,idx] = CartesianMesh.neighbors(mesh,idx)
    end
    return connectivity
end

function active_cells(poly::InterpolatingPolynomial{1,NF,B},coeffs) where
    {NF,B<:PolynomialBasis.AbstractBasis{dim}} where {dim}

    nf,ncells = size(coeffs)
    @assert nf == NF
    isactivecell = zeros(Bool,2,ncells)
    xL,xR = reference_cell(dim)
    box = IntervalBox(xL,xR)

    for idx in 1:ncells
        update!(poly,coeffs[:,idx])
        s = sign(poly,box)
        if s == 1
            isactivecell[1,idx] = true
        elseif s == -1
            isactivecell[2,idx] = true
        elseif s == 0
            isactivecell[[1,2],idx] .= true
        else
            throw(ArgumentError("Expected s ∈ {-1,0,+1}, got s = $s"))
        end
    end
    return isactivecell
end
