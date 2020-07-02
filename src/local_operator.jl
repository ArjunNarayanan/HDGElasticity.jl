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

function LLop(basis,quad)

    dim = dimension(basis)
    sdim = symmetric_tensor_dimension(dim)

    return mass_matrix(sdim,basis,quad)
end

function LLop(basis,quad,map)
    return determinant_jacobian(map)*LLop(basis,quad)
end

function LUop(basis,quad,Dhalf,Ek,map)

    NF = number_of_basis_functions(basis)
    sdim,n = size(Dhalf)
    @assert sdim == n
    @assert length(Ek) > 0

    dim = size(Ek[1])[2]
    @assert all([size(E) == (sdim,dim) for E in Ek])

    ALU = zeros(sdim*NF,dim*NF)
    invjac = inverse_jacobian(map)

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
    quad::QuadratureRule{dim},Dhalf::M,
    jac::AffineMap) where {dim,M<:AbstractMatrix}

    sdim = symmetric_tensor_dim(dim)
    Ek = vec_to_symm_mat_converter(dim)
    return get_stress_displacement_coupling(basis,quad,Dhalf,Ek,jac,dim,sdim)
end

function update_displacement_coupling!(AUU::Matrix,F::Function,
    surface_quad::QuadratureRule{1},jac,tau,dim)

    for (p,w) in surface_quad
        vals = F(p)
        N = interpolation_matrix(vals,dim)
        Ntau = tau*N
        AUU .+= N'*Ntau*jac*w
    end
end

function get_displacement_coupling(basis::TensorProductBasis{2,T,NF},
    surface_quad::QuadratureRule{1},
    jac::AffineMap,tau::R,x0,dx,dim) where {T,NF,R<:Real}

    nudofs = dim*NF
    AUU = zeros(nudofs,nudofs)

    update_displacement_coupling!(AUU,x->basis(extend(x,2,x0[2])),
        surface_quad,jac.jac[1],tau,dim)
    update_displacement_coupling!(AUU,x->basis(extend(x,1,x0[1]+dx[1])),
        surface_quad,jac.jac[2],tau,dim)
    update_displacement_coupling!(AUU,x->basis(extend(x,2,x0[2]+dx[2])),
        surface_quad,jac.jac[1],tau,dim)
    update_displacement_coupling!(AUU,x->basis(extend(x,1,x0[1])),
        surface_quad,jac.jac[2],tau,dim)

    return AUU
end

function get_displacement_coupling(basis::TensorProductBasis{2,T,NF},
    surface_quad::QuadratureRule{1},
    jac::AffineMap,tau::R,dim) where {T,NF,R<:Real}

    x0,dx = reference_element(basis)
    return get_displacement_coupling(basis,surface_quad,jac,tau,x0,dx,dim)
end

function get_displacement_coupling(basis::TensorProductBasis{dim,T,NF},
    surface_quad::QuadratureRule{1},
    jac::AffineMap,tau::R) where {dim,fdim,T,NF,R<:Real}

    x0,dx = reference_element(basis)
    return get_displacement_coupling(basis,surface_quad,jac,tau,x0,dx,dim)
end

function LocalOperator(basis::TensorProductBasis{dim},
    quad::QuadratureRule{dim},
    surface_quad::QuadratureRule{1},
    Dhalf,jac::AffineMap,tau) where {dim}

    if dim != 2
        throw(ArgumentError("Expected dim = 2, got dim = $dim"))
    end
    LL = -1.0*get_stress_coupling(basis,quad,jac)
    LU = get_stress_displacement_coupling(basis,quad,Dhalf,jac)
    UU = get_displacement_coupling(basis,surface_quad,jac,tau)
    return LocalOperator(LL,LU,UU)
end
