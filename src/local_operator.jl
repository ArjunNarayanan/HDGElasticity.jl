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

function AffineMapJacobian(jac::Vector{T},invjac::Vector{T},detjac::T) where {T}
    nj = length(jac)
    nij = length(invjac)
    @assert nj == nij
    sjac = SVector{nj}(jac)
    sinvjac = SVector{nij}(invjac)
    return AffineMapJacobian(sjac,sinvjac,detjac)
end

function AffineMapJacobian(element_size::T,reference_element_size::S) where {T<:AbstractVector,S<:Real}
    jac = element_size/reference_element_size
    invjac = inv.(jac)
    detjac = prod(jac)
    return AffineMapJacobian(jac,invjac,detjac)
end

function AffineMapJacobian(element_size::S,quad::TensorProductQuadratureRule{D,T}) where {S<:AbstractVector} where {D,T}
    reference_element_size = get_reference_element_size(T)
    return AffineMapJacobian(element_size,reference_element_size)
end

function AffineMapJacobian(mesh::UniformMesh,quad::TensorProductQuadratureRule)
    return AffineMapJacobian(mesh.element_size,quad)
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

function get_stress_displacement_coupling(basis::AbstractBasis{NF},
    quad::TensorProductQuadratureRule,Dhalf::AbstractMatrix,Ek::Vector{M},
    jac::AffineMapJacobian,dim,sdim) where {NF} where {M<:AbstractMatrix}

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

function get_stress_displacement_coupling(basis::TensorProductBasis{dim,T,NF},
    quad::TensorProductQuadratureRule{dim},Dhalf::AbstractMatrix,
    jac::AffineMapJacobian) where {dim,T,NF}

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

function reference_element(basis::TensorProductBasis{2})
    x0 = [-1.0,-1.0]
    dx = [2.0,2.0]
    return x0,dx
end

function reference_normals(basis::TensorProductBasis{2})
    return [[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]]
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

    LL = -1.0*get_stress_coupling(basis,quad,jac)
    LU = get_stress_displacement_coupling(basis,quad,Dhalf,jac)
    UU = get_displacement_coupling(basis,surface_quad,jac,tau)
    return LocalOperator(LL,LU,UU)
end

function LocalOperator(basis,quad,surface_quad,mesh,Dhalf,tau)
    jac = AffineMapJacobian(mesh)
    return LocalOperator(basis,quad,surface_quad,Dhalf,jac,tau)
end
