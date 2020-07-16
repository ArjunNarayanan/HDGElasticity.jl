struct SystemMatrix{T,Z}
    rows::Vector{Z}
    cols::Vector{Z}
    vals::Vector{T}
    function SystemMatrix(rows::Vector{Z},cols::Vector{Z},
        vals::Vector{T}) where {T<:Real,Z<:Integer}

        @assert length(rows) == length(cols)
        @assert length(cols) == length(vals)

        new{T,Z}(rows,cols,vals)
    end
end

function update!(matrix::SystemMatrix,rows,cols,vals)
    @assert length(rows) == length(cols)
    @assert length(cols) == length(vals)

    append!(matrix.rows,rows)
    append!(matrix.cols,cols)
    append!(matrix.vals,vals)
end

struct SystemRHS{T,Z}
    rows::Vector{Z}
    vals::Vector{T}
    function SystemRHS(rows::Vector{Z},vals::Vector{R}) where {Z<:Integer,R<:Real}
        @assert length(rows) == length(vals)
        new{R,Z}(rows,vals)
    end
end

function SystemRHS()
    Z = default_integer_type()
    R = default_float_type()
    rows = Z[]
    vals = R[]
    return SystemRHS(rows,vals)
end

function update!(rhs::SystemRHS,rows,vals)
    @assert length(rows) == length(vals)
    append!(rhs.rows,rows)
    append!(rhs.vals,vals)
end

function SystemMatrix()
    Z = default_integer_type()
    R = default_float_type()
    rows = Z[]
    cols = Z[]
    vals = R[]
    return SystemMatrix(rows,cols,vals)
end

function element_dof_start(elid,dofspernode,nodesperelement)
    edofs = dofspernode*nodesperelement
    return (elid-1)*edofs+1
end

function element_dof_stop(elid,dofspernode,nodesperelement)
    edofs = dofspernode*nodesperelement
    return elid*edofs
end

function element_dofs(elid,dofspernode,nodesperelement)
    start = element_dof_start(elid,dofspernode,nodesperelement)
    stop = element_dof_stop(elid,dofspernode,nodesperelement)
    return start:stop
end

function operator_dofs(row_dofs,col_dofs)
    lr = length(row_dofs)
    lc = length(col_dofs)
    rows = repeat(row_dofs,outer=lc)
    cols = repeat(col_dofs,inner=lr)
    return rows,cols
end

function assemble_local_operator!(system_matrix,opvals,elid,dim,NF)

    edofs = element_dofs(elid,dim,NF)
    rows,cols = operator_dofs(edofs,edofs)
    update!(system_matrix,rows,cols,opvals)
end

function assemble_traction_continuity!(system_matrix,llop,lhop,hhop,dgmesh)

    ncells = length(dgmesh.domain)


end
