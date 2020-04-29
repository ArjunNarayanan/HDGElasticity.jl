function check_all_matrix_sizes(a::Vector{Matrix{T}}) where {T}
    @assert length(a) > 0
    m,n = size(a[1])
    for matrix in a
        @assert size(matrix) == (m,n)
    end
end

struct LocalHybridCoupling{T}
    LUhat::Vector{Matrix{T}}
    UUhat::Vector{Matrix{T}}
    UhatUhat::Matrix{T}
    local_hybrid_operator::Vector{Matrix{T}}
    function LocalHybridCoupling(LUhat::Vector{Matrix{T}},
        UUhat::Vector{Matrix{T}},UhatUhat::Matrix{T}) where {T}

        check_all_matrix_sizes(LUhat)
        check_all_matrix_sizes(UUhat)

        nh = size(UhatUhat)[2]
        @assert all([size(m)[2] == nh for m in LUhat])
        @assert all([size(m)[2] == nh for m in UUhat])

        nfaces = length(LUhat)
        @assert length(UUhat) == nfaces

        local_hybrid_operator = [[LUhat[i];UUhat[i]] for i = 1:nfaces]

        new{T}(LUhat,UUhat,UhatUhat,local_hybrid_operator)
    end
end

function ALUhat_face1(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::AbstractMatrix,
    jac::AffineMapJacobian,
    sdim) where {T1,NF1,T2,NF2}

    dim1 = 2
    dim2 = 1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = -1.0*Ek[2]'*Dhalf
    J = jac.jac[1]

    for (p,w) in surface_quad
        vals = basis(p,-1.0)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function ALUhat_face2(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::AbstractMatrix,
    jac::AffineMapJacobian,
    sdim) where {T1,NF1,T2,NF2}

    dim1 = 2
    dim2 = 1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = 1.0*Ek[1]'*Dhalf
    J = jac.jac[2]

    for (p,w) in surface_quad
        vals = basis(1.0,p)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function ALUhat_face3(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::AbstractMatrix,
    jac::AffineMapJacobian,
    sdim) where {T1,NF1,T2,NF2}

    dim1 = 2
    dim2 = 1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = 1.0*Ek[2]'*Dhalf
    J = jac.jac[1]

    for (p,w) in surface_quad
        vals = basis(p,1.0)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function ALUhat_face4(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::AbstractMatrix,
    jac::AffineMapJacobian,
    sdim) where {T1,NF1,T2,NF2}

    dim1 = 2
    dim2 = 1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = -1.0*Ek[1]'*Dhalf
    J = jac.jac[2]

    for (p,w) in surface_quad
        vals = basis(-1.0,p)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function get_ALUhat(basis,surface_basis,surface_quad,Dhalf,jac)

    sdim = symmetric_tensor_dim(2)

    ALUhat = [ALUhat_face1(basis,surface_basis,surface_quad,Dhalf,jac,sdim),
              ALUhat_face2(basis,surface_basis,surface_quad,Dhalf,jac,sdim),
              ALUhat_face3(basis,surface_basis,surface_quad,Dhalf,jac,sdim),
              ALUhat_face4(basis,surface_basis,surface_quad,Dhalf,jac,sdim)]
    return ALUhat
end

function AUUhat_face1(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,tau,jac::AffineMapJacobian) where {T1,NF1,T2,NF2}

    dim = 2

    AUUhat = zeros(dim*NF1,dim*NF2)
    J = jac.jac[1]

    for (p,w) in surface_quad
        vals = basis(p,-1.0)
        svals = surface_basis(p)

        N = interpolation_matrix(vals,dim)
        Ntau = tau*interpolation_matrix(svals,dim)

        AUUhat += -N'*Ntau*J*w
    end
    return AUUhat
end

function AUUhat_face2(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,tau,jac::AffineMapJacobian) where {T1,NF1,T2,NF2}

    dim = 2

    AUUhat = zeros(dim*NF1,dim*NF2)
    J = jac.jac[2]

    for (p,w) in surface_quad
        vals = basis(1.0,p)
        svals = surface_basis(p)

        N = interpolation_matrix(vals,dim)
        Ntau = tau*interpolation_matrix(svals,dim)

        AUUhat += -N'*Ntau*J*w
    end
    return AUUhat
end

function AUUhat_face3(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,tau,jac::AffineMapJacobian) where {T1,NF1,T2,NF2}

    dim = 2

    AUUhat = zeros(dim*NF1,dim*NF2)
    J = jac.jac[1]

    for (p,w) in surface_quad
        vals = basis(p,1.0)
        svals = surface_basis(p)

        N = interpolation_matrix(vals,dim)
        Ntau = tau*interpolation_matrix(svals,dim)

        AUUhat += -N'*Ntau*J*w
    end
    return AUUhat
end

function AUUhat_face4(basis::TensorProductBasis{2,T1,NF1},
    surface_basis::TensorProductBasis{1,T2,NF2},
    surface_quad::ReferenceQuadratureRule,tau,jac::AffineMapJacobian) where {T1,NF1,T2,NF2}

    dim = 2

    AUUhat = zeros(dim*NF1,dim*NF2)
    J = jac.jac[2]

    for (p,w) in surface_quad
        vals = basis(-1.0,p)
        svals = surface_basis(p)

        N = interpolation_matrix(vals,dim)
        Ntau = tau*interpolation_matrix(svals,dim)

        AUUhat += -N'*Ntau*J*w
    end
    return AUUhat
end

function get_AUUhat(basis,surface_basis,surface_quad,tau,jac)
    AUUhat = [AUUhat_face1(basis,surface_basis,surface_quad,tau,jac),
              AUUhat_face2(basis,surface_basis,surface_quad,tau,jac),
              AUUhat_face3(basis,surface_basis,surface_quad,tau,jac),
              AUUhat_face4(basis,surface_basis,surface_quad,tau,jac)]
    return AUUhat
end

function get_AUhatUhat(surface_basis::TensorProductBasis{1,T,NF},
    surface_quad::ReferenceQuadratureRule,tau,jac::AffineMapJacobian) where {T,NF}

    dim = 2
    AUhatUhat = zeros(dim*NF,dim*NF)
    J = jac.jac[1]

    for (p,w) in surface_quad
        svals = surface_basis(p)

        N = interpolation_matrix(svals,dim)

        AUhatUhat += N'*tau*N*J*w
    end
    return AUhatUhat
end


function LocalHybridCoupling(basis,surface_basis,surface_quad,mesh,Dhalf,tau)
    jac = AffineMapJacobian(mesh)
    LUhat = get_ALUhat(basis,surface_basis,surface_quad,Dhalf,jac)
    UUhat = get_AUUhat(basis,surface_basis,surface_quad,tau,jac)
    UhatUhat = get_AUhatUhat(surface_basis,surface_quad,tau,jac)
    return LocalHybridCoupling(LUhat,UUhat,UhatUhat)
end
