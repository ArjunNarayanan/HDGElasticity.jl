function check_lengths(rows::Vector{S},cols::Vector{S},vals::Vector{T}) where {T,S}
    lr = length(rows)
    lc = length(cols)
    lv = length(vals)
    @assert lr == lc
    @assert lc == lv
end

struct SystemMatrix{T}
    rows::Vector{Int64}
    cols::Vector{Int64}
    vals::Vector{T}
    function SystemMatrix(rows::Vector{Int64},cols::Vector{Int64},
        vals::Vector{T}) where {T}

        check_lengths(rows,cols,vals)

        new{T}(rows,cols,vals)
    end
end

function update!(matrix::SystemMatrix,rows,cols,vals)
    check_lengths(rows,cols,vals)
    append!(matrix.rows,rows)
    append!(matrix.cols,cols)
    append!(matrix.vals,vals)
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

function element_dofs_to_operator_dofs(row_dofs,col_dofs)
    lr = length(row_dofs)
    lc = length(col_dofs)
    rows = repeat(row_dofs,outer=lc)
    cols = repeat(col_dofs,inner=lr)
    return rows,cols
end

function hybrid_dof_start(hid,total_element_dofs,dim,NF)
    return total_element_dofs+(hid-1)*dim*NF+1
end

function hybrid_dof_stop(hid,total_element_dofs,dim,NF)
    return total_element_dofs+hid*dim*NF
end

function hybrid_dofs(hid,total_element_dofs,dim,NF)
    return hybrid_dof_start(hid,total_element_dofs,dim,NF):hybrid_dof_stop(hid,total_element_dofs,dim,NF)
end
