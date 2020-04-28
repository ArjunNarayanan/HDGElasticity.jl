function plane_strain_voigt_hooke_matrix_2d(lambda,mu)
    l2mu = lambda+2.0*mu
    matrix = [l2mu    lambda  0.0
              lambda  l2mu    0.0
              0.0     0.0     mu]
    return matrix
end

function plane_strain_voigt_hooke_matrix(lambda,mu,dim)
    if dim == 2
        return plane_strain_voigt_hooke_matrix_2d(lambda,mu)
    elseif dim == 3
        error("3D not implemented yet!")
    else
        error("Expected dim ∈ {1,2} got dim = $dim")
    end
end
