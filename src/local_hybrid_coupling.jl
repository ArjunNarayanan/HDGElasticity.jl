function check_all_matrix_sizes(a::Vector{Matrix{T}}) where {T}
    @assert length(a) > 0
    m,n = size(a[1])
    for matrix in a
        @assert size(matrix) == (m,n)
    end
end

struct LocalHybridCoupling{T}
    LH::Vector{Matrix{T}}
    UH::Vector{Matrix{T}}
    HH::Vector{Matrix{T}}
    function LocalHybridCoupling(LH::Vector{Matrix{T}},
        UH::Vector{Matrix{T}},HH::Vector{Matrix{T}}) where {T}

        check_all_matrix_sizes(LH)
        check_all_matrix_sizes(UH)
        check_all_matrix_sizes(HH)

        nh = size(HH[1])[1]

        @assert all([size(m)[2] == nh for m in LH])
        @assert all([size(m)[2] == nh for m in UH])

        nfaces = length(LH)
        @assert length(UH) == nfaces
        @assert length(HH) == nfaces

        new{T}(LH,UH,HH)
    end
end


function update_stress_hybrid_coupling!(LH::Matrix,F::Function,
    surface_basis::TensorProductBasis{1},
    surface_quad::TensorProductQuadratureRule{1},M::Matrix,jac,dim)

    for (p,w) in surface_quad
        vals = F(p)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,M)
        N = interpolation_matrix(svals,dim)

        LH .+= Mk'*N*jac*w
    end
end

function get_stress_hybrid_coupling(F::Function,surface_basis::TensorProductBasis{1},
    surface_quad::TensorProductQuadratureRule{1},
    normal,Dhalf,jac,dim,sdim,NF,NHF)

    @assert size(Dhalf) == (sdim,sdim)
    @assert length(normal) == dim
    Ek = vec_to_symm_mat_converter(dim)
    LH = zeros(sdim*NF,dim*NHF)

    for k = 1:dim
        M = normal[k]*Ek[k]'*Dhalf
        update_stress_hybrid_coupling!(LH,F,surface_basis,surface_quad,M,jac,dim)
    end
    return LH
end

function get_stress_hybrid_coupling(basis::TensorProductBasis{dim,T,NF},
    surface_basis::TensorProductBasis{1,T2,NHF},
    surface_quad::TensorProductQuadratureRule{1},Dhalf,jac::AffineMapJacobian,
    x0,dx,normals) where {dim,T,NF,T2,NHF}

    sdim = symmetric_tensor_dim(dim)

    LH1 = get_stress_hybrid_coupling(x->basis(extend(x,2,x0[2])),
        surface_basis,surface_quad,normals[1],Dhalf,jac.jac[1],dim,sdim,NF,NHF)
    LH2 = get_stress_hybrid_coupling(x->basis(extend(x,1,x0[1]+dx[1])),
        surface_basis,surface_quad,normals[2],Dhalf,jac.jac[2],dim,sdim,NF,NHF)
    LH3 = get_stress_hybrid_coupling(x->basis(extend(x,2,x0[2]+dx[2])),
        surface_basis,surface_quad,normals[3],Dhalf,-jac.jac[1],dim,sdim,NF,NHF)
    LH4 = get_stress_hybrid_coupling(x->basis(extend(x,1,x0[1])),
        surface_basis,surface_quad,normals[4],Dhalf,-jac.jac[2],dim,sdim,NF,NHF)
    return [LH1,LH2,LH3,LH4]

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

    UH = zeros(dim*NF,dim*NHF)
    for (p,w) in surface_quad
        vals = F(p)
        svals = surface_basis(p)

        N = interpolation_matrix(vals,dim)
        Ntau = tau*interpolation_matrix(svals,dim)

        UH += N'*Ntau*jac*w
    end
    return UH
end

function get_displacement_hybrid_coupling(basis::TensorProductBasis{dim,T,NF},
    surface_basis::TensorProductBasis{1,T2,NHF},surface_quad,tau,
    jac::AffineMapJacobian,x0,dx) where {dim,T,NF,T2,NHF}

    @assert length(x0) == dim
    @assert length(dx) == dim

    UH1 = get_displacement_hybrid_coupling(x->basis(extend(x,2,x0[2])),
        surface_basis,surface_quad,tau,jac.jac[1],dim,NF,NHF)
    UH2 = get_displacement_hybrid_coupling(x->basis(extend(x,1,x0[1]+dx[1])),
        surface_basis,surface_quad,tau,jac.jac[2],dim,NF,NHF)
    UH3 = get_displacement_hybrid_coupling(x->basis(extend(x,2,x0[2]+dx[2])),
        surface_basis,surface_quad,tau,-jac.jac[1],dim,NF,NHF)
    UH4 = get_displacement_hybrid_coupling(x->basis(extend(x,1,x0[1])),
        surface_basis,surface_quad,tau,-jac.jac[2],dim,NF,NHF)

    return [UH1,UH2,UH3,UH4]
end

function get_displacement_hybrid_coupling(basis,surface_basis,surface_quad,
    tau,jac)

    x0,dx = reference_element(basis)
    return get_displacement_hybrid_coupling(basis,surface_basis,surface_quad,
        tau,jac,x0,dx)
end

function get_hybrid_coupling(surface_basis::TensorProductBasis{1},
    surface_quad::TensorProductQuadratureRule{1},tau,jac,dim,NHF)

    HH = zeros(dim*NHF,dim*NHF)
    for (p,w) in surface_quad
        svals = surface_basis(p)

        N = interpolation_matrix(svals,dim)
        Ntau = tau*N

        HH .+= N'*Ntau*jac*w
    end
    return HH
end

function get_hybrid_coupling(surface_basis::TensorProductBasis{1,T,NHF},
    surface_quad::TensorProductQuadratureRule{1},
    tau,jac::AffineMapJacobian) where {T,NHF}

    dim = 2
    HH1 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        jac.jac[1],dim,NHF)
    HH2 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        jac.jac[2],dim,NHF)
    HH3 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        -jac.jac[1],dim,NHF)
    HH4 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        -jac.jac[2],dim,NHF)

    return [HH1,HH2,HH3,HH4]
end

function LocalHybridCoupling(basis,surface_basis,surface_quad,Dhalf,tau,jac)
    LH = get_stress_hybrid_coupling(basis,surface_basis,surface_quad,Dhalf,jac)
    UH = get_displacement_hybrid_coupling(basis,surface_basis,surface_quad,tau,jac)
    HH = get_hybrid_coupling(surface_basis,surface_quad,tau,jac)
    return LocalHybridCoupling(LH,UH,HH)
end
