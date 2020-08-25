# struct LocalOperator{T}
#     LL::Matrix{T}
#     LU::Matrix{T}
#     UU::Matrix{T}
#     local_operator::Matrix{T}
#     lulop
#     function LocalOperator(LL::Matrix{T},LU::Matrix{T},UU::Matrix{T}) where {T}
#         llm,lln = size(LL)
#         lum,lun = size(LU)
#         uum,uun = size(UU)
#         @assert llm == lln
#         @assert uum == uun
#         @assert llm == lum
#         @assert lun == uum
#
#         lop = [LL   LU
#                LU'  UU]
#
#         lulop = lu(lop)
#         new{T}(LL,LU,UU,lop,lulop)
#     end
# end

function LLop(basis,quad)

    dim = dimension(basis)
    sdim = symmetric_tensor_dimension(dim)

    return mass_matrix(basis,quad,-1.0,sdim)
end

function LLop(basis,quad,cellmap)
    return determinant_jacobian(cellmap)*LLop(basis,quad)
end

function LUop(basis,grad_basis,quad,ED,cellmap,NF)

    @assert length(ED) > 0

    dim,sdim = size(ED[1])
    @assert all([size(E) == (dim,sdim) for E in ED])

    ALU = zeros(sdim*NF,dim*NF)
    invjac = inverse_jacobian(cellmap)
    detjac = determinant_jacobian(cellmap)

    for (p,w) in quad
        vals = basis(p)
        grads = grad_basis(p)
        for k = 1:length(ED)
            gradsk = invjac[k]*grads[:,k]
            Mk = make_row_matrix(gradsk,ED[k])
            N = interpolation_matrix(vals,dim)

            ALU .+= Mk'*N*detjac*w
        end
    end

    return ALU
end

function LUop(basis,quad,Dhalf,cellmap)
    dim = dimension(basis)
    NF = number_of_basis_functions(basis)
    Ek = vec_to_symm_mat_converter(dim)
    ED = [E'*Dhalf for E in Ek]
    return LUop(basis,x->gradient(basis,x),quad,ED,cellmap,NF)
end

function UUop(basis,facequads,facemaps,cellmap,stabilization)
    dim = dimension(basis)
    scale = face_determinant_jacobian(cellmap)
    return stabilization*mass_matrix_on_boundary(basis,facequads,facemaps,
        scale,dim)
end

function UUop(basis,facequads,facemaps,iquad,normals,imap,cellmap,
    stabilization,ndofs,NF)

    facescale = face_determinant_jacobian(cellmap)
    iscale = scale_area(cellmap,normals)
    return stabilization*mass_matrix_on_boundary(basis,facequads,facemaps,
        facescale,iquad,imap,iscale,ndofs,NF)

end

function UUop(basis,facequads,facemaps,iquad,normals,imap,
    cellmap,stabilization)

    dim = dimension(basis)
    nf = number_of_basis_functions(basis)

    return UUop(basis,facequads,facemaps,iquad,normals,imap,cellmap,
        stabilization,dim,nf)
end

function local_operator(LL,LU,UU)
    lop = [LL   LU
           LU'  UU]
    return lop
end

function local_operator(basis,vquad,facequad,facemaps,Dhalf,
    cellmap,stabilization)

    LL = LLop(basis,vquad,cellmap)
    LU = LUop(basis,vquad,Dhalf,cellmap)
    UU = UUop(basis,facequad,facemaps,cellmap,stabilization)
    return local_operator(LL,LU,UU)
end

function local_operator(basis,vquad,facequads,facemaps,iquad,normals,imap,
    Dhalf,cellmap,stabilization)

    LL = LLop(basis,vquad,cellmap)
    LU = LUop(basis,vquad,Dhalf,cellmap)
    UU = UUop(basis,facequads,facemaps,iquad,normals,imap,
        cellmap,stabilization)
    return local_operator(LL,LU,UU)
end
