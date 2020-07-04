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

    return mass_matrix(basis,quad,sdim,1.0)
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


function update_UUop!(AUU,func,facequad,detjac::T,
    stabilization,ndofs) where {T<:Real}

    for (p,w) in facequad
        vals = func(p)
        N = interpolation_matrix(vals,ndofs)
        AUU .+= detjac*stabilization*N'*N*w
    end

end

function update_UUop!(AUU,basis,isactiveface,facequads,map::AffineMap,
    stabilization,ndofs)

    dim = dimension(basis)
    cell = reference_cell(dim)

    funcs = restrict_on_faces(basis,cell)
    jac = jacobian(map,cell)

    for (faceid,f) in enumerate(funcs)
        if isactiveface[faceid]
            update_UUop!(AUU,f,facequads[faceid],jac[faceid],stabilization,ndofs)
        end
    end

end

function update_UUop!(AUU,basis,iquad,imap::InterpolatingPolynomial,
    stabilization,ndofs)

    for (p,w) in iquad
        vals = basis(imap(p))
        detjac = determinant_jacobian(imap,p)
        N = interpolation_matrix(vals,ndofs)
        AUU .+= detjac*stabilization*N'*N*w
    end

end

function UUop(basis,isactiveface,facequads,iquad,cellmap,imap,stabilization,ndofs)

    dim = dimension(basis)
    NF = number_of_basis_functions(basis)
    cell = reference_cell(dim)

    nudofs = ndofs*NF
    AUU = zeros(nudofs,nudofs)

    update_UUop!(AUU,basis,isactiveface,facequads,cellmap,stabilization,ndofs)
    update_UUop!(AUU,basis,iquad,imap,stabilization,ndofs)

    return AUU

end

function UUop(basis,facequad,map,stabilization,ndofs)

    dim = dimension(basis)
    NF = number_of_basis_functions(basis)
    cell = reference_cell(dim)

    nudofs = ndofs*NF
    AUU = zeros(nudofs,nudofs)

    funcs = restrict_on_faces(basis,cell)
    jac = jacobian(map,cell)

    for (faceid,f) in enumerate(funcs)
        update_UUop!(AUU,f,facequad,jac[faceid],stabilization,ndofs)
    end

    return AUU
end

function UUop(basis,facequad,map,stabilization)
    dim = dimension(basis)
    return UUop(basis,facequad,map,stabilization,dim)
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
