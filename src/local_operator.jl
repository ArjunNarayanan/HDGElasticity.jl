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

function jacobian(element_size::SVector{2},quadrature_weights::SVector{N}) where {N}
    return prod(element_size)/sum(quadrature_weights)
end

function jacobian(mesh::UniformMesh{2},quad::TensorProductQuadratureRule{2})
    return jacobian(mesh.element_size,quad.weights)
end

function make_row_matrix(vals::AbstractVector,matrix::AbstractMatrix)
    return hcat([v*matrix for v in vals]...)
end

function interpolation_matrix(vals::AbstractVector,dim::Int64)
    return make_row_matrix(vals,diagm(ones(dim)))
end

function vec_to_symm_mat_converter(dim::Int64)
    if dim == 2
        E1 = @SMatrix [1.0 0.0
                       0.0 0.0
                       0.0 1.0]
        E2 = @SMatrix [0.0 0.0
                       0.0 1.0
                       1.0 0.0]
        return [E1,E2]
    elseif dim == 3
        E1 = @SMatrix [1.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   1.0   0.0
                       0.0   0.0   1.0
                       0.0   0.0   0.0]
        E2 = @SMatrix [0.0   0.0   0.0
                       0.0   1.0   0.0
                       0.0   0.0   0.0
                       1.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   0.0   1.0]
        E3 = @SMatrix [0.0   0.0   0.0
                       0.0   0.0   0.0
                       0.0   0.0   1.0
                       0.0   0.0   0.0
                       1.0   0.0   0.0
                       0.0   1.0   0.0]
        return [E1,E2,E3]
    else
        throw(ArgumentError("Expected dim ∈ {1,2} got dim = $dim"))
    end
end

function symmetric_tensor_dim(dim::Int64)
    if dim == 2
        return 3
    elseif dim == 3
        return 6
    else
        throw(ArgumentError("Expected dim ∈ {2,3}, got dim = $dim"))
    end
end

struct AffineMapJacobian{dim,T}
    jac::SVector{dim,T}
    invjac::SVector{dim,T}
    detjac::T
end

function AffineMapJacobian(element_size::T,reference_element_size::S) where {T<:AbstractVector,S<:Real}
    jac = element_size/reference_element_size
    invjac = inv.(jac)
    detjac = prod(jac)
    return AffineMapJacobian(jac,invjac,detjac)
end

function AffineMapJacobian(element_size,quad::TensorProductQuadratureRule{D,T}) where {D,T}
    reference_element_size = get_reference_element_size(T)
    return AffineMapJacobian(element_size,reference_element_size)
end

function AffineMapJacobian(mesh::UniformMesh)
    return AffineMapJacobian(mesh.element_size)
end

function get_LMass(basis::TensorProductBasis{dim,T,NF},
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

function get_LUStiffness(basis::TensorProductBasis{dim,T,NF},
    quad::TensorProductQuadratureRule,Dhalf::AbstractMatrix,jac::AffineMapJacobian,sdim) where {dim,T,NF}

    m,n = size(Dhalf)
    @assert m == n && m == sdim

    Ek = vec_to_symm_mat_converter(dim)
    ALU = zeros(sdim*NF,dim*NF)

    for k = 1:dim
        E = Ek[k]
        ED = E'*Dhalf
        invjac = jac.invjac[k]
        for (p,w) in quad
            dvals = invjac*gradient(basis,k,p)
            vals = basis(p)
            Mk = make_row_matrix(vec(dvals),ED)
            N = interpolation_matrix(vals,dim)

            ALU += Mk'*N*jac.detjac*w
        end
    end
    return ALU
end

function get_UMass(basis::TensorProductBasis{2,T,NF},
    surface_quad::TensorProductQuadratureRule{1},
    jac::AffineMapJacobian,tau::Float64) where {T,NF}

    dim = 2
    nudofs = dim*NF
    AUU = zeros(nudofs,nudofs)

    for (pvec,w) in surface_quad
        p = pvec[1]
        vals = basis(p,-1.0)
        N = interpolation_matrix(vals,dim)
        Ntau = tau*N
        AUU += N'*Ntau*jac.jac[1]*w

        vals = basis(1.0,p)
        N = interpolation_matrix(vals,dim)
        Ntau = tau*N
        AUU += N'*Ntau*jac.jac[2]*w

        vals = basis(p,1.0)
        N = interpolation_matrix(vals,dim)
        Ntau = tau*N
        AUU += N'*Ntau*jac.jac[1]*w

        vals = basis(-1.0,p)
        N = interpolation_matrix(vals,dim)
        Ntau = tau*N
        AUU += N'*Ntau*jac.jac[2]*w
    end
    return AUU
end

function LocalOperator(basis::TensorProductBasis{dim},
    quad::TensorProductQuadratureRule{dim},
    surface_quad::TensorProductQuadratureRule{1},
    Dhalf,jac::AffineMapJacobian,tau) where {dim}

    sdim = symmetric_tensor_dim(dim)

    ALL = -1.0*get_LMass(basis,quad,jac,sdim)
    ALU = get_LUStiffness(basis,quad,Dhalf,jac,sdim)
    AUU = get_UMass(basis,surface_quad,jac,tau)
    return LocalOperator(ALL,ALU,AUU)
end

function LocalOperator(basis,quad,surface_quad,mesh,Dhalf,tau)
    jac = AffineMapJacobian(mesh)
    return LocalOperator(basis,quad,surface_quad,Dhalf,jac,tau)
end
