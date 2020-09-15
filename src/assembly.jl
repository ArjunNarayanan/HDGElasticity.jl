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

function SystemMatrix()
    Z = default_integer_type()
    R = default_float_type()
    rows = Z[]
    cols = Z[]
    vals = R[]
    return SystemMatrix(rows,cols,vals)
end

function assemble!(matrix::SystemMatrix,rows,cols,vals)
    @assert length(rows) == length(cols)
    @assert length(cols) == length(vals)

    append!(matrix.rows,rows)
    append!(matrix.cols,cols)
    append!(matrix.vals,vals)
end

struct SystemRHS{T,Z}
    rows::Vector{Z}
    vals::Vector{T}
    function SystemRHS(rows::Vector{Z},vals::Vector{T}) where {Z<:Integer,T<:Real}
        @assert length(rows) == length(vals)
        new{T,Z}(rows,vals)
    end
end

function SystemRHS()
    Z = default_integer_type()
    R = default_float_type()
    rows = Z[]
    vals = R[]
    return SystemRHS(rows,vals)
end

function assemble!(rhs::SystemRHS,rows,vals)
    @assert length(rows) == length(vals)
    append!(rhs.rows,rows)
    append!(rhs.vals,vals)
end

function element_dof_start(elid,dofsperelement)
    return (elid-1)*dofsperelement+1
end

function element_dof_stop(elid,dofsperelement)
    return elid*dofsperelement
end

function element_dofs(elid,dofsperelement)
    start = element_dof_start(elid,dofsperelement)
    stop = element_dof_stop(elid,dofsperelement)
    return start:stop
end

function operator_dofs(row_dofs,col_dofs)
    lr = length(row_dofs)
    lc = length(col_dofs)
    rows = repeat(row_dofs,outer=lc)
    cols = repeat(col_dofs,inner=lr)
    return rows,cols
end

function assemble!(system_matrix::SystemMatrix,vals,rowelids,
    colelids,dofsperelement)

    rowdofs = vcat([element_dofs(r,dofsperelement) for r in rowelids]...)
    coldofs = vcat([element_dofs(c,dofsperelement) for c in colelids]...)

    oprows,opcols = operator_dofs(rowdofs,coldofs)
    assemble!(system_matrix,oprows,opcols,vals)
end

function assemble!(system_rhs::SystemRHS,vals,rowelid,dofsperelement)
    rowdofs = element_dofs(rowelid,dofsperelement)
    assemble!(system_rhs,rowdofs,vals)
end

function assemble_displacement_face!(system_matrix::SystemMatrix,HH,
    elid,dofsperelement)

    assemble!(system_matrix,vec(HH),elid,elid,dofsperelement)
end

function assemble_traction_face!(system_matrix::SystemMatrix,HL,iLLxLH,HH,rowelid,
    colelids,dofsperelement)

    tractionop = HL*iLLxLH
    assemble!(system_matrix,tractionop,rowelid,colelids,dofsperelement)
    assemble!(system_matrix,-HH,rowelid,rowelid,dofsperelement)
end

function SparseArrays.sparse(system_matrix::SystemMatrix,ndofs)
    return sparse(system_matrix.rows,system_matrix.cols,system_matrix.vals,
        ndofs,ndofs)
end

function rhs(system_rhs::SystemRHS,ndofs)
    return Array(sparsevec(system_rhs.rows,system_rhs.vals,ndofs))
end
