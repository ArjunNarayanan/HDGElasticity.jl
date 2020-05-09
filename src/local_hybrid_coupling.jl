function check_all_matrix_sizes(a::Vector{Matrix{T}}) where {T}
    @assert length(a) > 0
    m,n = size(a[1])
    for matrix in a
        @assert size(matrix) == (m,n)
    end
end

struct LocalHybridCoupling{T}
    LUh::Vector{Matrix{T}}
    UUh::Vector{Matrix{T}}
    UhUh::Matrix{T}
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


function update_stress_hybrid_coupling!(LUh::Matrix,F::Function,
    surface_basis::TensorProductBasis{1},
    surface_quad::TensorProductQuadratureRule{1},M::Matrix,jac,dim)

    for (p,w) in surface_quad
        vals = F(p)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,M)
        N = interpolation_matrix(svals,dim)

        LUh .+= Mk'*N*jac*w
    end
end

function get_stress_hybrid_coupling(F::Function,surface_basis::TensorProductBasis{1},
    surface_quad::TensorProductQuadratureRule{1},
    normal,Dhalf,jac,dim,sdim,NF,NHF)

    @assert size(Dhalf) == (sdim,sdim)
    @assert length(normal) == dim
    Ek = vec_to_symm_mat_converter(dim)
    LUh = zeros(sdim*NF,dim*NHF)

    for k = 1:dim
        M = normal[k]*Ek[k]'*Dhalf
        update_stress_hybrid_coupling!(LUh,F,surface_basis,surface_quad,M,jac,dim)
    end
    return LUh
end

function get_stress_hybrid_coupling(basis::TensorProductBasis{dim,T,NF},
    surface_basis::TensorProductBasis{1,T2,NHF},
    surface_quad::TensorProductQuadratureRule{1},Dhalf,jac::AffineMapJacobian,
    x0,dx,normals) where {dim,T,NF,T2,NHF}

    sdim = symmetric_tensor_dim(dim)

    LU1 = get_stress_hybrid_coupling(x->basis(extend(x,2,x0[2])),
        surface_basis,surface_quad,normals[1],Dhalf,jac.jac[1],dim,sdim,NF,NHF)
    LU2 = get_stress_hybrid_coupling(x->basis(extend(x,1,x0[1]+dx[1])),
        surface_basis,surface_quad,normals[2],Dhalf,jac.jac[2],dim,sdim,NF,NHF)
    LU3 = get_stress_hybrid_coupling(x->basis(extend(x,2,x0[2]+dx[2])),
        surface_basis,surface_quad,normals[3],Dhalf,-jac.jac[1],dim,sdim,NF,NHF)
    LU4 = get_stress_hybrid_coupling(x->basis(extend(x,1,x0[1])),
        surface_basis,surface_quad,normals[4],Dhalf,-jac.jac[2],dim,sdim,NF,NHF)
    return [LU1,LU2,LU3,LU4]

end

function get_stress_hybrid_coupling(basis,surface_basis,surface_quad,Dhalf,jac)
    x0,dx = reference_element(basis)
    normals = reference_normals(basis)
    return get_stress_hybrid_coupling(basis,surface_basis,surface_quad,
        Dhalf,jac,x0,dx,normals)
end

function get_displacement_hybrid_coupling(F::Function,
    surface_basis::TensorProductBasis{1},
    surface_quad::TensorProductQuadratureRule{1},tau,jac,dim,NF,NHF)

    UUh = zeros(dim*NF,dim*NHF)
    for (p,w) in surface_quad
        vals = F(p)
        svals = surface_basis(p)

        N = interpolation_matrix(vals,dim)
        Ntau = tau*interpolation_matrix(svals,dim)

        UUh += N'*Ntau*jac*w
    end
    return UUh
end

function get_displacement_hybrid_coupling(basis::TensorProductBasis{dim,T,NF},
    surface_basis::TensorProductBasis{1,T2,NHF},surface_quad,tau,
    jac::AffineMapJacobian,x0,dx) where {dim,T,NF,T2,NHF}

    @assert length(x0) == dim
    @assert length(dx) == dim

    UU1 = get_displacement_hybrid_coupling(x->basis(extend(x,2,x0[2])),
        surface_basis,surface_quad,tau,jac.jac[1],dim,NF,NHF)
    UU2 = get_displacement_hybrid_coupling(x->basis(extend(x,1,x0[1]+dx[1])),
        surface_basis,surface_quad,tau,jac.jac[2],dim,NF,NHF)
    UU3 = get_displacement_hybrid_coupling(x->basis(extend(x,2,x0[2]+dx[2])),
        surface_basis,surface_quad,tau,-jac.jac[1],dim,NF,NHF)
    UU4 = get_displacement_hybrid_coupling(x->basis(extend(x,1,x0[1])),
        surface_basis,surface_quad,tau,-jac.jac[2],dim,NF,NHF)

    return [UU1,UU2,UU3,UU4]
end

function get_displacement_hybrid_coupling(basis,surface_basis,surface_quad,
    tau,jac)

    x0,dx = reference_element(basis)
    return get_displacement_hybrid_coupling(basis,surface_basis,surface_quad,
        tau,jac,x0,dx)

end

function get_AUhatUhat(surface_basis::TensorProductBasis{1,T,NF},
    surface_quad::TensorProductQuadratureRule{1},tau,jac::AffineMapJacobian) where {T,NF}

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
