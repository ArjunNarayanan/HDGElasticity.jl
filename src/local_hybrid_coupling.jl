struct LocalHybridCoupling
    LUhat::Matrix
    UUhat::Matrix
end

function ALUhat_face1(basis::TensorProductBasis{dim1,T1,NF1},
    surface_basis::TensorProductBasis{dim2,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::Matrix,
    jac::AffineMapJacobian,
    sdim) where {dim1,T1,NF1,dim2,T2,NF2}

    @assert dim1 == dim2+1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = -1.0*Ek[2]'*Dhalf
    J = jac.jac[1]

    for (p,w) in surface_quad
        vals = basis(p,-1.0)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function ALUhat_face2(basis::TensorProductBasis{dim1,T1,NF1},
    surface_basis::TensorProductBasis{dim2,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::Matrix,
    jac::AffineMapJacobian,
    sdim) where {dim1,T1,NF1,dim2,T2,NF2}

    @assert dim1 == dim2+1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = 1.0*Ek[1]'*Dhalf
    J = jac.jac[2]

    for (p,w) in surface_quad
        vals = basis(1.0,p)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function ALUhat_face3(basis::TensorProductBasis{dim1,T1,NF1},
    surface_basis::TensorProductBasis{dim2,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::Matrix,
    jac::AffineMapJacobian,
    sdim) where {dim1,T1,NF1,dim2,T2,NF2}

    @assert dim1 == dim2+1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = 1.0*Ek[2]'*Dhalf
    J = jac.jac[1]

    for (p,w) in surface_quad
        vals = basis(p,1.0)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function ALUhat_face4(basis::TensorProductBasis{dim1,T1,NF1},
    surface_basis::TensorProductBasis{dim2,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::Matrix,
    jac::AffineMapJacobian,
    sdim) where {dim1,T1,NF1,dim2,T2,NF2}

    @assert dim1 == dim2+1

    dim = dim1
    Ek = vec_to_symm_mat_converter(dim)

    ALUhat = zeros(sdim*NF1,dim*NF2)

    ED = -1.0*Ek[1]'*Dhalf
    J = jac.jac[2]

    for (p,w) in surface_quad
        vals = basis(-1.0,p)
        svals = surface_basis(p)

        Mk = make_row_matrix(vals,ED)
        Nhat = interpolation_matrix(svals,dim1)

        ALUhat += -Mk'*Nhat*J*w
    end
    return ALUhat
end

function get_ALUhat(basis::TensorProductBasis{dim1,T1,NF1},
    surface_basis::TensorProductBasis{dim2,T2,NF2},
    surface_quad::ReferenceQuadratureRule,Dhalf::Matrix,
    jac::AffineMapJacobian) where {dim1,T1,NF1,dim2,T2,NF2}

    sdim = symmetric_tensor_dim(dim1)

    ALUhat = [ALUhat_face1(basis,surface_basis,surface_quad,Dhalf,jac,sdim),
              ALUhat_face2(basis,surface_basis,surface_quad,Dhalf,jac,sdim),
              ALUhat_face3(basis,surface_basis,surface_quad,Dhalf,jac,sdim),
              ALUhat_face4(basis,surface_basis,surface_quad,Dhalf,jac,sdim)]
    return ALUhat
end
