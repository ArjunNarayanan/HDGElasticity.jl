struct LocalOperator{T}
    LL::Matrix{T}
    LU::Matrix{T}
    UU::Matrix{T}
    local_operator::Matrix{T}
    lulop
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

        lulop = lu(lop)
        new{T}(LL,LU,UU,lop,lulop)
    end
end

function LLop(basis,quad)

    dim = dimension(basis)
    sdim = symmetric_tensor_dimension(dim)

    return mass_matrix(basis,quad,sdim,-1.0)
end

function LLop(basis,quad,cellmap)
    return determinant_jacobian(cellmap)*LLop(basis,quad)
end

function LUop(basis,quad,Dhalf,cellmap,Ek)

    NF = number_of_basis_functions(basis)
    sdim,n = size(Dhalf)
    @assert sdim == n
    @assert length(Ek) > 0

    dim = size(Ek[1])[2]
    @assert all([size(E) == (sdim,dim) for E in Ek])

    ALU = zeros(sdim*NF,dim*NF)
    invjac = inverse_jacobian(cellmap)
    detjac = determinant_jacobian(cellmap)

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

function LUop(basis,quad,Dhalf,cellmap)
    dim = dimension(basis)
    Ek = vec_to_symm_mat_converter(dim)
    return LUop(basis,quad,Dhalf,cellmap,Ek)
end

# function UUop(basis,facequads,isactiveface,cellmap,stabilization)
#     dim = dimension(basis)
#     return stabilization*mass_matrix_on_boundary(basis,facequads,
#         isactiveface,dim,cellmap)
# end

function UUop(basis,facequad,cellmap,stabilization)

    dim = dimension(basis)
    return stabilization*mass_matrix_on_boundary(basis,facequad,dim,cellmap)
end

function UUop(basis,facequads,isactiveface,iquad,normals,imap,
    cellmap,stabilization)

    dim = dimension(basis)
    return stabilization*mass_matrix_on_boundary(basis,facequads,isactiveface,
        iquad,normals,imap,dim,cellmap)
end

function LocalOperator(basis,vquad,facequad,Dhalf,cellmap,stabilization)

    LL = LLop(basis,vquad,cellmap)
    LU = LUop(basis,vquad,Dhalf,cellmap)
    UU = UUop(basis,facequad,cellmap,stabilization)
    return LocalOperator(LL,LU,UU)
end

# function LocalOperator(basis,vquad,facequads,isactiveface,
#         Dhalf,cellmap,stabilization)
#
#     LL = LLop(basis,vquad,cellmap)
#     LU = LUop(basis,vquad,Dhalf,cellmap)
#     UU = UUop(basis,facequads,isactiveface,cellmap,stabilization)
#     return LocalOperator(LL,LU,UU)
# end

function LocalOperator(basis,vquad,facequads,isactiveface,iquad,normals,imap,
    Dhalf,cellmap,stabilization)

    LL = LLop(basis,vquad,cellmap)
    LU = LUop(basis,vquad,Dhalf,cellmap)
    UU = UUop(basis,facequads,isactiveface,iquad,normals,imap,
        cellmap,stabilization)
    return LocalOperator(LL,LU,UU)
end
