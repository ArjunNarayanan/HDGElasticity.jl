using Test
using IntervalArithmetic
using PolynomialBasis
using CartesianMesh
using Revise
using HDGElasticity


function allequal(v1,v2)
    return all(v1 .== v2)
end

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

x0 = [0.,0.]
widths = [2.,1.]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)

domain = HDGElasticity.cell_domain(mesh)
testdomain = [IntervalBox(0..1,0..1),IntervalBox(1..2,0..1)]
@test allequal(domain,testdomain)

connectivity = HDGElasticity.cell_connectivity(mesh)
testconn = [(0,0) (0,0)
            (2,4) (0,0)
            (0,0) (0,0)
            (0,0) (1,2)]
@test all([allequal(connectivity[i],testconn[i]) for i = 1:8])

bgmesh = HDGElasticity.BackgroundMesh(mesh)
@test allequal(bgmesh.domain,testdomain)
@test all([allequal(bgmesh.connectivity[i],testconn[i]) for i = 1:8])
