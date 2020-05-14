function check_lengths(rows::Vector{S},cols::Vector{S},vals::Vector{T}) where {T<:Real,S<:Int}
    lr = length(rows)
    lc = length(cols)
    lv = length(vals)
    @assert lr == lc
    @assert lc == lv
    return true
end

struct SystemMatrix{T}
    rows::Vector{Int64}
    cols::Vector{Int64}
    vals::Vector{T}
    function SystemMatrix(rows::Vector{Int64},cols::Vector{Int64},
        vals::Vector{T}) where {T<:Real}

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
    function SystemRHS(rows::Vector{Z},vals::Vector{R}) where {Z<:Integer,R<:Real}
        @assert length(rows) == length(vals)
        new{R}(rows,vals)
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

function element_stress_dofs(elid,dim,sdim,NF)
    dofs = Int[]

    start = element_dof_start(elid,dim,sdim,NF)
    stop = start+sdim-1
    step = dim+sdim
    for F in 1:NF
        append!(dofs,start:stop)
        start = start+step
        stop = start+sdim-1
    end
    return dofs
end

function element_dofs_to_operator_dofs(row_dofs,col_dofs)
    lr = length(row_dofs)
    lc = length(col_dofs)
    rows = repeat(row_dofs,outer=lc)
    cols = repeat(col_dofs,inner=lr)
    return rows,cols
end

function hybrid_dof_start(hid,total_element_dofs,dim,NHF)
    return total_element_dofs+(hid-1)*dim*NHF+1
end

function hybrid_dof_stop(hid,total_element_dofs,dim,NF)
    return total_element_dofs+hid*dim*NF
end

function hybrid_dofs(hid,total_element_dofs,dim,NF)
    return hybrid_dof_start(hid,total_element_dofs,dim,NF):hybrid_dof_stop(hid,total_element_dofs,dim,NF)
end

function assemble_local_operator!(matrix::SystemMatrix{T},
    vec_local_operator_vals::Vector{T},
    elid::S,dim::S,sdim::S,NF::S) where {T<:Real,S<:Integer}

    edofs = element_dofs(elid,dim,sdim,NF)
    rows,cols = element_dofs_to_operator_dofs(edofs,edofs)
    update!(matrix,rows,cols,vec_local_operator_vals)

end

function assemble_local_operator!(matrix::SystemMatrix{T},
    vec_local_operator_vals::Vector{T},
    elids::V,dim::S,sdim::S,NF::S) where {T<:Real,S<:Integer,V<:AbstractVector}

    for elid in elids
        assemble_local_operator!(matrix,vec_local_operator_vals,
            elid,dim,sdim,NF)
    end
end

function assemble_local_hybrid_operator!(matrix::SystemMatrix{R},
    vec_local_hybrid_operator_vals::Vector{Vector{R}},
    face_to_hybrid_element_number::Matrix{Z},edofs::V,elid::Z,faceid::Z,
    total_element_dofs::Z,dim::Z,NHF::Z) where {R<:Real,Z<:Integer,V<:AbstractVector}

    hid = face_to_hybrid_element_number[faceid,elid]
    hdofs = hybrid_dofs(hid,total_element_dofs,dim,NHF)
    rows,cols = element_dofs_to_operator_dofs(edofs,hdofs)
    update!(matrix,rows,cols,vec_local_hybrid_operator_vals[faceid])

end

function assemble_local_hybrid_operator!(matrix,vec_local_hybrid_operator_vals,
    face_to_hybrid_element_number,edofs,elid,faceids::V,
    total_element_dofs,dim,NHF) where {V<:AbstractVector}

    for faceid in faceids
        assemble_local_hybrid_operator!(matrix,vec_local_hybrid_operator_vals,
            face_to_hybrid_element_number,edofs,elid,faceid,total_element_dofs,
            dim,NHF)
    end
end

function assemble_local_hybrid_operator!(matrix,vec_local_hybrid_operator_vals,
    face_to_hybrid_element_number,elids::V,faceids::V,
    total_element_dofs,dim,sdim,NF,NHF) where {V<:AbstractVector}

    for elid in elids
        edofs = element_dofs(elid,dim,sdim,NF)
        assemble_local_hybrid_operator!(matrix,vec_local_hybrid_operator_vals,
            face_to_hybrid_element_number,edofs,elid,faceids,total_element_dofs,
            dim,NHF)
    end

end

function assemble_hybrid_local_operator!(matrix::SystemMatrix{R},
    vec_hybrid_local_operator_vals::Vector{Vector{R}},
    face_to_hybrid_element_number::Matrix{Z},edofs::V,elid::Z,faceid::Z,
    total_element_dofs::Z,dim::Z,NHF::Z) where {R<:Real,Z<:Integer,V<:AbstractVector}

    hid = face_to_hybrid_element_number[faceid,elid]
    hdofs = hybrid_dofs(hid,total_element_dofs,dim,NHF)
    rows,cols = element_dofs_to_operator_dofs(hdofs,edofs)
    update!(matrix,rows,cols,vec_hybrid_local_operator_vals[faceid])

end

function assemble_hybrid_local_operator!(matrix,vec_hybrid_local_operator_vals,
    face_to_hybrid_element_number,edofs,elid,faceids::V,
    total_element_dofs,dim,NHF) where {V<:AbstractVector}

    for faceid in faceids
        assemble_hybrid_local_operator!(matrix,vec_hybrid_local_operator_vals,
            face_to_hybrid_element_number,edofs,elid,faceid,total_element_dofs,
            dim,NHF)
    end

end

function assemble_hybrid_local_operator!(matrix,vec_hybrid_local_operator_vals,
    face_to_hybrid_element_number,elids::V1,faceids::V2,total_element_dofs,
    dim,sdim,NF,NHF) where {V1<:AbstractVector,V2<:AbstractVector}

    for elid in elids
        edofs = element_dofs(elid,dim,sdim,NF)
        assemble_hybrid_local_operator!(matrix,vec_hybrid_local_operator_vals,
            face_to_hybrid_element_number,edofs,elid,faceids,total_element_dofs,
            dim,NHF)
    end

end

function assemble_hybrid_local_operator!(matrix,operator_vals,
    mesh::UniformMesh{dim},total_element_dofs,NF,NHF) where {dim}

    sdim = symmetric_tensor_dim(dim)
    for elid in 1:mesh.total_number_of_elements
        edofs = element_dofs(elid,dim,sdim,NF)
        faceids = findall(mesh.face_indicator[:,elid] .== :internal)
        assemble_hybrid_local_operator!(matrix,operator_vals,
            mesh.face_to_hybrid_element_number,edofs,elid,faceids,
            total_element_dofs,dim,NHF)
    end

end

function assemble_hybrid_operator!(matrix::SystemMatrix{R},
    vec_hybrid_operator_vals::Vector{Vector{R}},
    face_to_hybrid_element_number::Matrix{Z},elid::Z,faceid::Z,
    total_element_dofs::Z,dim::Z,NHF::Z) where {R<:Real,Z<:Integer,V<:AbstractVector}

    hid = face_to_hybrid_element_number[faceid,elid]
    hdofs = hybrid_dofs(hid,total_element_dofs,dim,NHF)
    rows,cols = element_dofs_to_operator_dofs(hdofs,hdofs)
    update!(matrix,rows,cols,vec_hybrid_operator_vals[faceid])

end

function assemble_hybrid_operator!(matrix,vec_hybrid_operator_vals,
    face_to_hybrid_element_number,elid,faceids::V,
    total_element_dofs,dim,NHF) where {V<:AbstractVector}

    for faceid in faceids
        assemble_hybrid_operator!(matrix,vec_hybrid_operator_vals,
            face_to_hybrid_element_number,elid,faceid,total_element_dofs,
            dim,NHF)
    end

end

function assemble_hybrid_operator!(matrix,vec_hybrid_operator_vals,
    face_to_hybrid_element_number,elids::V,faceids::V,total_element_dofs,
    dim,NHF) where {V<:AbstractVector}

    for elid in elids
        assemble_hybrid_operator!(matrix,vec_hybrid_operator_vals,
            face_to_hybrid_element_number,elid,faceids,total_element_dofs,
            dim,NHF)
    end

end

function assemble_hybrid_operator!(matrix,operator_vals,mesh::UniformMesh{dim},
    total_element_dofs,NHF) where {dim}

    sdim = symmetric_tensor_dim(dim)
    for elid in 1:mesh.total_number_of_elements
        faceids = findall(mesh.face_indicator[:,elid] .== :internal)
        assemble_hybrid_operator!(matrix,operator_vals,
            mesh.face_to_hybrid_element_number,elid,faceids,
            total_element_dofs,dim,NHF)
    end

end
