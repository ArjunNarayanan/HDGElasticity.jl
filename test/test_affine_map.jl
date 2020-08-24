using Test
using StaticArrays
using IntervalArithmetic
using Revise
using HDGElasticity

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

@test HDGElasticity.linear_map_slope(1.,2.,[1.,2.],[3.,4.]) ≈ [2.,2.]

map = HDGElasticity.LineMap(0.,1.,[0.,0.],[1.,1.])
@test map.xiL ≈ 0.0
@test map.xiR ≈ 1.0
@test allapprox(map.xL,[0.,0.])
@test allapprox(map.xR,[1.,1.])
@test allapprox(map.slope,[1.,1.])
@test HDGElasticity.dimension(map) == 2
@test allapprox(map(0.),[0.,0.])
@test allapprox(map(0.5),[0.5,0.5])

map = HDGElasticity.LineMap([0.5,0.5],[1.,0.5])
@test map.xiL ≈ -1.0
@test map.xiR ≈ 1.0
@test allapprox(map.xL,[0.5,0.5])
@test allapprox(map.xR,[1.0,0.5])
@test allapprox(map.slope,[0.5,0.0]/2)

map = HDGElasticity.LineMap([0.,0.],[1.,1.]*sqrt(2))
@test allapprox(map(-1),[0.,0.])
@test allapprox(map(1),[1,1]*sqrt(2))
@test allapprox(HDGElasticity.jacobian(map),[1,1]/sqrt(2))
@test allapprox(HDGElasticity.determinant_jacobian(map),1.)
@test allapprox(HDGElasticity.inverse_jacobian(map),[1,1]*sqrt(2))

xiL = [0.,0.]
xiR = [1.,1.]
xL = [0.,0.]
xR = [2.,1.]
map = HDGElasticity.CellMap(xiL,xiR,xL,xR)
@test allapprox(map.xiL,xiL)
@test allapprox(map.xiR,xiR)
@test allapprox(map.xL,xL)
@test allapprox(map.xR,xR)
@test HDGElasticity.dimension(map) == 2
@test allapprox(map([0.5,0.5]),[1.,0.5])
@test allapprox(map([0.25,0.75]),[0.5,0.75])
@test allapprox(HDGElasticity.jacobian(map),[2.,1.])
@test allapprox(HDGElasticity.inverse_jacobian(map),[0.5,1.])
@test HDGElasticity.determinant_jacobian(map) ≈ 2.0
@test allapprox(HDGElasticity.face_determinant_jacobian(map),[2.,1.,2.,1.])

HDGElasticity.update_range!(map,[1.,1.],[2.,2.])
@test allapprox(map([0.5,0.5]),[1.5,1.5])
@test allapprox(map([0.75,0.25]),[1.75,1.25])
@test allapprox(HDGElasticity.jacobian(map),[1.,1.])
@test HDGElasticity.determinant_jacobian(map) ≈ 1.0

map = HDGElasticity.CellMap(2)
@test allapprox(map.xiL,[-1.,-1.])
@test allapprox(map.xiR,[+1.,+1.])
@test allapprox(map.xL,[-1.,-1.])
@test allapprox(map.xR,[+1.,+1.])
@test allapprox(map.slope,[1.,1.])
HDGElasticity.update_range!(map,[0.,0.],[1.,1.])
@test allapprox(map.xL,[0.,0.])
@test allapprox(map.xR,[1.,1.])
@test allapprox(map.slope,[0.5,0.5])

box = IntervalBox(1..2,3..5)
HDGElasticity.update_range!(map,box)
@test allapprox(map.xL,[1.,3.])
@test allapprox(map.xR,[2.,5.])
@test allapprox(map.slope,[0.5,1.])

map = HDGElasticity.CellMap([1.,2.],[2.,4.])
@test allapprox(map.xiL,[-1.,-1.])
@test allapprox(map.xiR,[1.,1.])
@test allapprox(map.xL,[1.,2.])
@test allapprox(map.xR,[2.,4.])
@test allapprox(HDGElasticity.jacobian(map),[0.5,1.])
@test HDGElasticity.determinant_jacobian(map) ≈ 0.5

xi = IntervalBox(-1..1,2)
x = IntervalBox(0..1,2)
map = HDGElasticity.CellMap(xi,x)
@test allapprox(map.xiL,[-1.,-1.])
@test allapprox(map.xiR,[1.,1.])
@test allapprox(map.xL,[0.,0.])
@test allapprox(map.xR,[1.,1.])

map = HDGElasticity.CellMap(IntervalBox(0..2,1..2))
@test allapprox(map.xiL,[-1.,-1.])
@test allapprox(map.xiR,[1.,1.])
@test allapprox(map.xL,[0.,1.])
@test allapprox(map.xR,[2.,2.])


fmaps = HDGElasticity.reference_cell_facemaps(2)
@test allapprox(fmaps[1](-1.0),[-1.,-1.])
@test allapprox(fmaps[1](+1.0),[+1.,-1.])

@test allapprox(fmaps[2](-1.0),[+1.,-1.])
@test allapprox(fmaps[2](+1.0),[+1.,+1.])

@test allapprox(fmaps[3](-1.0),[-1.,+1.])
@test allapprox(fmaps[3](+1.0),[+1.,+1.])

@test allapprox(fmaps[4](-1.0),[-1.,-1.])
@test allapprox(fmaps[4](+1.0),[-1.,+1.])
