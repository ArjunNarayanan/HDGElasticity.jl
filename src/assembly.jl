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

function update!(rhs::SystemRHS,rows,vals)
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

function assemble!(system_matrix,vals,rowelid::Z,
    colelid::Z,dofsperelement) where {Z<:Integer}

    rowdofs = element_dofs(rowelid,dofsperelement)
    coldofs = element_dofs(colelid,dofsperelement)

    oprows,opcols = operator_dofs(rowdofs,coldofs)
    update!(system_matrix,oprows,opcols,vals)
end

function assemble!(system_matrix,vals,rowelid::Z,
    colelids::V,dofsperelement) where {Z<:Integer,V<:AbstractVector}

    rowdofs = element_dofs(rowelid,dofsperelement)
    coldofs = vcat([element_dofs(c,dofsperelement) for c in colelids]...)

    oprows,opcols = operator_dofs(rowdofs,coldofs)
    update!(system_matrix,oprows,opcols,vals)
end

function assemble_local_operator!(system_matrix,opvals,elid,dim,NF)

    edofs = element_dofs(elid,dim,NF)
    rows,cols = operator_dofs(edofs,edofs)
    update!(system_matrix,rows,cols,opvals)
end

function assemble_traction_continuity!(system_matrix,llop,lhop,hhop,dgmesh,ufs)

    nface,nphase,ncells = size(dgmesh.isactiveface)
    ncells = length(dgmesh.domain)

    dim = dimension(ufs.vbasis)
    NHF = number_of_basis_functions(ufs.sbasis)

    for cellid in 1:ncells
        for phaseid in 1:nphase
            K = llop[phaseid,cellid].lulop
            for faceid in 1:nface
                nbrelid,nbrfaceid = dgmesh.connectivity[faceid,phaseid,cellid]

                if nbrelid != 0
                    LH = lhop[faceid,phaseid,cellid]
                    opvals = LH'*(K\LH) - hhop[faceid,phaseid,cellid]

                    hid = dgmesh.face2hid[faceid,phaseid,cellid]
                    hdofs = element_dofs(hid,dim,NHF)

                    rows,cols = operator_dofs(hdofs,hdofs)

                    update!(system_matrix,rows,cols,vec(opvals))
                end
            end
        end
    end

end
