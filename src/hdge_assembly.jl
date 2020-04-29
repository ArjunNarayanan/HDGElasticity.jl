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

struct SystemRHS{T}
    rows::Vector{Int64}
    vals::Vector{T}
    function SystemRHS(rows::Vector{Int64},vals::Vector{T}) where {T}
        @assert length(rows) == length(vals)
        new{T}(rows,vals)
    end
end

function SystemRHS()
    rows = Int[]
    vals = Float64[]
    return SystemRHS(rows,vals)
end

function update!(rhs::SystemRHS,rows,vals)
    @assert length(rows) == length(vals)
    append!(rhs.rows,rows)
    append!(rhs.vals,vals)
end

function SystemMatrix()
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]
    return SystemMatrix(rows,cols,vals)
end

function get_dofs_per_element(dim,sdim,NF)
    return (dim+sdim)*NF
end

function element_dof_start(elid,dim,sdim,NF)
    edofs = get_dofs_per_element(dim,sdim,NF)
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

function assemble_local_operator!(matrix::SystemMatrix{T},
    vec_local_operator_vals::Vector{T},
    total_nelements::Int64,dim::Int64,sdim::Int64,NF::Int64) where {T}

    for elid in 1:total_nelements
        edofs = element_dofs(elid,dim,sdim,NF)
        rows,cols = element_dofs_to_operator_dofs(edofs,edofs)
        update!(matrix,rows,cols,vec_local_operator_vals)
    end
end

function assemble_local_hybrid_operator!(matrix::SystemMatrix{T},
    vec_local_hybrid_operator_vals::Vector{Vector{T}},
    face_to_hybrid_element_number::Matrix{Int64},
    total_nelements::Int64,faces_per_element::Int64,total_element_dofs::Int64,
    dim::Int64,sdim::Int64,
    NF::Int64,NHF::Int64) where {T}

    for elid in 1:total_nelements
        edofs = element_dofs(elid,dim,sdim,NF)
        for faceid in 1:faces_per_element
            hid = face_to_hybrid_element_number[faceid,elid]
            hdofs = hybrid_dofs(hid,total_element_dofs,dim,NHF)
            rows,cols = element_dofs_to_operator_dofs(edofs,hdofs)
            update!(matrix,rows,cols,vec_local_hybrid_operator_vals[faceid])
        end
    end
end

function assemble_hybrid_local_operator!(matrix::SystemMatrix{T},
    vec_hybrid_local_operator_vals::Vector{Vector{T}},
    face_to_hybrid_element_number::Matrix{Int64},
    face_indicator::Matrix{Symbol},total_nelements::Int64,
    faces_per_element::Int64,total_element_dofs::Int64,
    dim::Int64,sdim::Int64,NF::Int64,NHF::Int64) where {T}

    for elid in 1:total_nelements
        edofs = element_dofs(elid,dim,sdim,NF)
        for faceid in 1:faces_per_element
            if face_indicator[faceid,elid] == :interior
                hid = face_to_hybrid_element_number[faceid,elid]
                hdofs = hybrid_dofs(hid,total_element_dofs,dim,NHF)
                rows,cols = element_dofs_to_operator_dofs(hdofs,edofs)
                update!(matrix,rows,cols,vec_hybrid_local_operator_vals[faceid])
            end
        end
    end
end

function assemble_hybrid_mass_operator!(matrix::SystemMatrix{T},
    vec_hybrid_mass_operator_vals::Vector{T},
    face_to_hybrid_element_number::Matrix{Int64},
    face_indicator::Matrix{Symbol},total_nelements::Int64,
    faces_per_element::Int64,total_element_dofs::Int64,
    dim::Int64,sdim::Int64,NHF::Int64) where {T}

    for elid in 1:total_nelements
        for faceid in 1:faces_per_element
            if face_indicator[faceid,elid] == :interior
                hid = face_to_hybrid_element_number[faceid,elid]
                hdofs = hybrid_dofs(hid,total_element_dofs,dim,NHF)
                rows,cols = element_dofs_to_operator_dofs(hdofs,hdofs)
                update!(matrix,rows,cols,vec_hybrid_mass_operator_vals)
            end
        end
    end

end
