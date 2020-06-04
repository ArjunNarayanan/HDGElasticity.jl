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

function nodal_coordinates(mesh::UniformMesh{dim,FT},
    basis::TensorProductBasis{dim,T,NF}) where {dim,T,NF,FT}

    ncells = mesh.total_number_of_elements
    coords = Matrix{FT}(undef,dim,NF*ncells)
    start = 1
    stop = NF
    for idx in 1:ncells
        xL,xR = CartesianMesh.element(mesh,idx)
        map = HDGElasticity.AffineMap(xL,xR)
        coords[:,start:stop] = map(basis.points)
        start = stop+1
        stop += NF
    end
    return coords
end

function distance_function(coords,xc)
    return coords[1,:] .- xc
end

x0 = [0.,0.]
widths = [2.,1.]
nelements = [2,1]
mesh = UniformMesh(x0,widths,nelements)

domain = HDGElasticity.cell_domain(mesh)
testdomain = [IntervalBox(0..1,0..1),IntervalBox(1..2,0..1)]
@test allequal(domain,testdomain)

connectivity = HDGElasticity.cell_connectivity(mesh)
testconn = [0  0
            2  0
            0  0
            0  1]
@test allequal(connectivity,testconn)

basis = TensorProductBasis(2,1)
poly = InterpolatingPolynomial(1,basis)
NF = HDGElasticity.number_of_basis_functions(basis)
coords = nodal_coordinates(mesh,basis)
xc = 0.75
coeffs = reshape(distance_function(coords,xc),NF,:)
isactivecell = HDGElasticity.active_cells(poly,coeffs)
testactivecells = [true true
                   true false]
@test allequal(isactivecell,testactivecells)

f(x) = poly(x,-1.0)
s = sign(f,IntervalBox(-1..1),5,1e-2)
