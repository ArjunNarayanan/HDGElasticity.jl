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

function LUop(basis,quad,Dhalf,map,Ek)

    NF = number_of_basis_functions(basis)
    sdim,n = size(Dhalf)
    @assert sdim == n
    @assert length(Ek) > 0

    dim = size(Ek[1])[2]
    @assert all([size(E) == (sdim,dim) for E in Ek])

    ALU = zeros(sdim*NF,dim*NF)
    invjac = inverse_jacobian(map)
    detjac = determinant_jacobian(map)

    for k = 1:length(Ek)
        E = Ek[k]
        ED = E'*Dhalf
        for (p,w) in quad
            dvals = invjac[k]*gradient(basis,k,p)
            vals = basis(p)
            Mk = make_row_matrix(dvals,ED)
            N = interpolation_matrix(vals,dim)

            ALU += Mk'*N*detjac*w
        end
    end
    return ALU
end

function LUop(basis,quad,Dhalf,map)
    dim = dimension(basis)
    Ek = vec_to_symm_mat_converter(dim)
    return LUop(basis,quad,Dhalf,map,Ek)
end


function update_LUUop!(AUU,func,surface_quad,detjac,stabilization,ndofs)
    for (p,w) in surface_quad
        vals = func(p)
        N = interpolation_matrix(vals,ndofs)
        AUU .+= stabilization*N'*N*detjac*w
    end
end

function get_displacement_coupling(basis,surface_quad,map,stabilization)

    dim = dimension(basis)
    NF = number_of_basis_functions(basis)
    cell = reference_cell(dim)

    ndofs = dim*NF
    AUU = zeros(ndofs,ndofs)

    funcs = restrict_on_faces(basis,cell)

    update_LUUop!(AUU,x->basis(extend(x,2,x0[2])),
        surface_quad,jac.jac[1],tau,dim)
    update_LUUop!(AUU,x->basis(extend(x,1,x0[1]+dx[1])),
        surface_quad,jac.jac[2],tau,dim)
    update_LUUop!(AUU,x->basis(extend(x,2,x0[2]+dx[2])),
        surface_quad,jac.jac[1],tau,dim)
    update_LUUop!(AUU,x->basis(extend(x,1,x0[1])),
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
