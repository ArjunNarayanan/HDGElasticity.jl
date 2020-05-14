struct LocalOperator{T}
    LL::Matrix{T}
    LU::Matrix{T}
    UU::Matrix{T}
    local_operator::Matrix{T}
    function LocalOperator(LL::Matrix{T},LU::Matrix{T},UU::Matrix{T}) where {T}
        llm,lln = size(LL)
        lum,lun = size(LU)
        uum,uun = size(UU)
        @assert llm == lln
        @assert uum == uun
        @assert llm == lum
        @assert lun == uum

        lop = [LL   LU
               LU'  UU]

        new{T}(LL,LU,UU,lop)
    end
end

function get_stress_coupling(basis::TensorProductBasis{dim,T,NF},
    quad::TensorProductQuadratureRule,jac::AffineMapJacobian,sdim) where {dim,T,NF}

    nldofs = sdim*NF
    ALL = zeros(nldofs,nldofs)

    for (p,w) in quad
        vals = basis(p)
        M = interpolation_matrix(vals,sdim)
        ALL += M'*M*jac.detjac*w
    end
    return ALL
end

function get_stress_coupling(basis::TensorProductBasis{dim},
    quad::TensorProductQuadratureRule{dim},jac::AffineMapJacobian) where {dim}

    sdim = symmetric_tensor_dim(dim)
    return get_stress_coupling(basis,quad,jac,sdim)
end

function get_stress_displacement_coupling(basis::TensorProductBasis{D,T,NF},
    quad::TensorProductQuadratureRule,Dhalf::M1,Ek::Vector{M2},
    jac::AffineMapJacobian,dim,sdim) where {D,T,NF} where {M1<:AbstractMatrix,M2<:AbstractMatrix}

    m,n = size(Dhalf)
    @assert m == n && m == sdim
    @assert all([size(E) == (sdim,dim) for E in Ek])

    ALU = zeros(sdim*NF,dim*NF)

    for k = 1:length(Ek)
        E = Ek[k]
        ED = E'*Dhalf
        invjac = jac.invjac[k]
        for (p,w) in quad
            dvals = invjac*gradient(basis,k,p)
            vals = basis(p)
            Mk = make_row_matrix(dvals,ED)
            N = interpolation_matrix(vals,dim)

            ALU += Mk'*N*jac.detjac*w
        end
    end
    return ALU
end

function get_stress_displacement_coupling(basis::TensorProductBasis{dim},
    quad::TensorProductQuadratureRule{dim},Dhalf::M,
    jac::AffineMapJacobian) where {dim,M<:AbstractMatrix}

    sdim = symmetric_tensor_dim(dim)
    Ek = vec_to_symm_mat_converter(dim)
    return get_stress_displacement_coupling(basis,quad,Dhalf,Ek,jac,dim,sdim)
end

function update_displacement_coupling!(AUU::Matrix,F::Function,
    surface_quad::TensorProductQuadratureRule{1},jac,tau,dim)

    for (p,w) in surface_quad
        vals = F(p)
        N = interpolation_matrix(vals,dim)
        Ntau = tau*N
        AUU .+= N'*Ntau*jac*w
    end
end

function get_displacement_coupling(basis::TensorProductBasis{2,T,NF},
    surface_quad::TensorProductQuadratureRule{1},
    jac::AffineMapJacobian,tau::Float64,x0,dx,dim) where {T,NF}

    nudofs = dim*NF
    AUU = zeros(nudofs,nudofs)

    update_displacement_coupling!(AUU,x->basis(extend(x,2,x0[2])),
        surface_quad,jac.jac[1],tau,dim)
    update_displacement_coupling!(AUU,x->basis(extend(x,1,x0[1]+dx[1])),
        surface_quad,jac.jac[2],tau,dim)
    update_displacement_coupling!(AUU,x->basis(extend(x,2,x0[2]+dx[2])),
        surface_quad,-jac.jac[1],tau,dim)
    update_displacement_coupling!(AUU,x->basis(extend(x,1,x0[1])),
        surface_quad,-jac.jac[2],tau,dim)

    return AUU
end

function get_displacement_coupling(basis::TensorProductBasis{2,T,NF},
    surface_quad::TensorProductQuadratureRule{1},
    jac::AffineMapJacobian,tau::Float64,dim) where {T,NF}

    x0,dx = reference_element(basis)
    return get_displacement_coupling(basis,surface_quad,jac,tau,x0,dx,dim)
end

function get_displacement_coupling(basis::TensorProductBasis{dim,T,NF},
    surface_quad::TensorProductQuadratureRule{1},
    jac::AffineMapJacobian,tau::Float64) where {dim,fdim,T,NF}

    x0,dx = reference_element(basis)
    return get_displacement_coupling(basis,surface_quad,jac,tau,x0,dx,dim)
end

function LocalOperator(basis::TensorProductBasis{dim},
    quad::TensorProductQuadratureRule{dim},
    surface_quad::TensorProductQuadratureRule{1},
    Dhalf,jac::AffineMapJacobian,tau) where {dim}

    if dim != 2
        throw(ArgumentError("Expected dim = 2, got dim = $dim"))
    end
    LL = -1.0*get_stress_coupling(basis,quad,jac)
    LU = get_stress_displacement_coupling(basis,quad,Dhalf,jac)
    UU = get_displacement_coupling(basis,surface_quad,jac,tau)
    return LocalOperator(LL,LU,UU)
end
