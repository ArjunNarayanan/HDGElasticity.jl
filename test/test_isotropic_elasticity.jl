using Test
using HDGElasticity

l = 2.0
m = 3.0

matrix = HDGElasticity.plane_strain_voigt_hooke_matrix_2d(l,m)
testmatrix = [(l+2m)  l     0.0
                l   (l+2m) 0.0
               0.0   0.0    m]
@test all(matrix .== testmatrix)

matrix = HDGElasticity.plane_strain_voigt_hooke_matrix(l,m,2)
@test all(matrix .== testmatrix)
@test_throws ArgumentError HDGElasticity.plane_strain_voigt_hooke_matrix(l,m,3)
@test_throws ArgumentError HDGElasticity.plane_strain_voigt_hooke_matrix(l,m,1)
