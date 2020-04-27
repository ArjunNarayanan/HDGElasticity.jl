struct SystemMatrix{T}
    rows::Vector{Int64}
    cols::Vector{Int64}
    vals::Vector{T}
    function SystemMatrix(rows::Vector{Int64},cols::Vector{Int64},
        vals::Vector{T}) where {T}

        lr = length(rows)
        lc = length(cols)
        lv = length(vals)
        @assert lr == lc
        @assert lc == lv

        new{T}(rows,cols,vals)
    end
end

function SystemMatrix()
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]
    return SystemMatrix(rows,cols,vals)
end

function element_dof_start(elid,dim,sdim,NF)
    edofs = (dim+sdim)*NF
    return (elid-1)*edofs+1
end

function element_dof_stop(elid,dim,sdim,NF)
    edofs = (dim+sdim)*NF
    return elid*edofs
end

function element_dofs(elid,dim,sdim,NF)
    return element_dof_start(elid,dim,sdim,NF):element_dof_stop(elid,dim,sdim,NF)
end

function element_dofs_to_operator_dofs(dofs)
    ldofs = length(dofs)
    rows = repeat(dofs,outer=ldofs)
    cols = repeat(dofs,inner=ldofs)
    return rows,cols
end
