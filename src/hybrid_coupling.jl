function get_hybrid_coupling(surface_basis::TensorProductBasis{1},
    surface_quad::QuadratureRule{1},tau,jac,dim,NHF)

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
    surface_quad::QuadratureRule{1},
    tau,jac) where {T,NHF}

    dim = 2
    HH1 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        jac.jac[1],dim,NHF)
    HH2 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        jac.jac[2],dim,NHF)
    HH3 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        jac.jac[1],dim,NHF)
    HH4 = get_hybrid_coupling(surface_basis,surface_quad,tau,
        jac.jac[2],dim,NHF)

    return [HH1,HH2,HH3,HH4]
end
