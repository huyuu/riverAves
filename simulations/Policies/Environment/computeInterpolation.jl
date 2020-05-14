# using Pkg
# Pkg.add("CSV")
# Pkg.add("Interpolations")
using CSV
# reference: https://github.com/JuliaMath/Interpolations.jl/blob/master/doc/Interpolations.jl.ipynb
using Interpolations

# MARK: - Models

struct Boundary
    value::Float64
    f::Function
    type::Char
end
Boundary(; value::Number, f::Function, type::Char) = let
    Boundary(float(value), f, type)
end



# MARK: - Compute Functions

function solve3DEllipticPDE(; xyLowerB::Boundary, xyUpperB::Boundary, yzLowerB::Boundary, yzUpperB::Boundary, xzLowerB::Boundary, xzUpperB::Boundary, points::Int, error::Float64=1e-6)::Array{Float64, 3}
    xs = LinRange(yzLowerB.value, yzUpperB.value, points)
    hx = (yzUpperB.value - yzLowerB.value)/(points-1)
    ys = LinRange(xzLowerB.value, xzUpperB.value, points)
    hy = (xzUpperB.value - xzLowerB.value)/(points-1)
    zs = LinRange(xyLowerB.value, xyUpperB.value, points)
    hz = (xyUpperB.value - xyLowerB.value)/(points-1)
    fs::Array{Float64, 3} = zeros((points, points, points))
    # set Dirichlet Boundaries
    for i = 1:points, j = 1:points
        if xyLowerB.type == 'D'
            fs[i, j, 1] = xyLowerB.f(i, j, xyLowerB.value)
        end
        if xyUpperB.type == 'D'
            fs[i, j, end] = xyUpperB.f(i, j, xyUpperB.value)
        end
        if yzLowerB.type == 'D'
            fs[1, i, j] = yzLowerB.f(yzLowerB.value, i, j)
        end
        if yzUpperB.type == 'D'
            fs[end, i, j] = yzUpperB.f(yzUpperB.value, i, j)
        end
        if xzLowerB.type == 'D'
            fs[i, 1, j] = xzLowerB.f(i, xzLowerB.value, j)
        end
        if xzUpperB.type == 'D'
            fs[i, end, j] = xzUpperB.f(i, xzUpperB.value, j)
        end
    end
    # main calculation
    iterCount::Int = 0
    while true
        maxDelta::Float64 = 0.0
        # update Neumann Boundarties
        for i = 1:points, j = 1:points
            if xyLowerB.type == 'N'
                fs[i, j, 1] = fs[i, j, 2] - hx * xyLowerB.f(xs[i], ys[j], xyLowerB.value)
            end
            if xyUpperB.type == 'N'
                fs[i, j, end] = fs[i, j, end-1] + hx * xyUpperB.f(xs[i], ys[j], xyUpperB.value)
            end
            if yzLowerB.type == 'N'
                fs[1, i, j] = fs[2, i, j] - hy * yzLowerB.f(yzLowerB.value, ys[i], zs[j])
            end
            if yzUpperB.type == 'N'
                fs[end, i, j] = fs[end-1, i, j] + hy * yzUpperB.f(yzUpperB.value, ys[i], zs[j])
            end
            if xzLowerB.type == 'N'
                fs[i, 1, j] = fs[i, 2, j] - hz * xzLowerB.f(xs[i], xzLowerB.value, ys[j])
            end
            if xzUpperB.type == 'N'
                fs[i, end, j] = fs[i, end-1, j] + hz * xzUpperB.f(xs[i], xzUpperB.value, zs[j])
            end
        end
        # update inner points
        for i = 2:points-1, j = 2:points-1, k = 2:points-1
            newfs = 1.0/6.0 * ( fs[i-1, j, k] + fs[i+1, j, k] + fs[i, j-1, k] + fs[i, j+1, k] + fs[i, j, k-1] + fs[i, j, k+1] )
            delta = abs(newfs - fs[i, j, k])
            maxDelta = max(delta, maxDelta)
            fs[i, j, k] = newfs
        end
        # check delta
        iterCount += 1
        # println("IterCount = $iterCount, MaxDelta = $maxDelta")
        if maxDelta < error
            break
        end
    end
    return fs
end


function saveResults(; dirName::String, xs::LinRange{Float64}, ys::LinRange{Float64}, zs::LinRange{Float64}, results::Array{Float64, 3})::Array{String}
    filePaths::Array{String} = []
    # save x samples
    filePath = let
        path = "$dirName/xSamplePoints.csv"
        file = open(path, "w")
        for x in xs
            write(file, "$x\n")
        end
        close(file)
        path
    end
    push!(filePaths, filePath)
    # save y samples
    filePath = let
        path = "$dirName/ySamplePoints.csv"
        file = open(path, "w")
        for y in ys
            write(file, "$y\n")
        end
        close(file)
        path
    end
    push!(filePaths, filePath)
    # save z samples
    if lowerUnit == "mb"
        zs = zs .\ 1.0
    end
    filePath = let
        path = "$dirName/zSamplePoints.csv"
        file = open(path, "w")
        for z in zs
            write(file, "$z\n")
        end
        close(file)
        path
    end
    push!(filePaths, filePath)
    # save results
    for (zIndex, zValue) in enumerate(zs)
        filePath = "$dirName/zValue=$(round(zValue, sigdigits=4)).csv"
        open(filePath, "w") do file
            for y in ys
                write(file, ",$y")
            end
            write(file, "\n")
            for (xIndex, xValue) in enumerate(xs)
                write(file, "$xValue")
                for (yIndex, yValue) in enumerate(ys)
                    write(file, ",$(results[xIndex, yIndex, zIndex])")
                end
                write(file, "\n")
            end
            push!(filePaths, filePath)
        end
    end
    return filePaths
end



# MARK: - Main

# set file paths from arguments
# reference: https://stackoverflow.com/questions/21056991/access-command-line-arguments-in-julia
const lowerTimeLowerLocationFilePath = ARGS[1]
const lowerLevel, lowerUnit = let
    _level, unit, _ = split(split(lowerTimeLowerLocationFilePath, "/")[end], "_")
    if unit == "mb"
        level = parse(Float64, _level)
        level = 1.0/level
    else
        level = parse(Float64, _level)
    end
    (level, unit)
end
# const lowerLevel = parse(Int, split(split(lowerTimeLowerLocationFilePath, "/")[end], "_")[1])
# const lowerUnit = split(split(lowerTimeLowerLocationFilePath, "/")[end], "_")[2]

const lowerTimeUpperLocationFilePath = ARGS[2]

const upperTimeLowerLocationFilePath = ARGS[3]
const upperLevel, upperUnit = let
    _level, unit, _ = split(split(lowerTimeUpperLocationFilePath, "/")[end], "_")
    if unit == "mb"
        level = parse(Float64, _level)
        level = 1.0/level
    else
        level = parse(Float64, _level)
    end
    (level, unit)
end
# const upperLevel = parse(Int, split(split(lowerTimeUpperLocationFilePath, "/")[end], "_")[1])
# const upperUnit = split(split(lowerTimeUpperLocationFilePath, "/")[end], "_")[2]

const upperTimeUpperLocationFilePath = ARGS[4]

# get input array
# lowerTimeLowerLocation
const ll = let
    data = []
    open(ARGS[1], "r") do file
        # reference: https://discourse.julialang.org/t/how-to-convert-dataframe-into-array-t-2/22487
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        data = CSV.read(file)[:, 2:end] |> Array{Float64, 2}
    end
    data
end
# lowerTimeUpperLocation
const lu = let
    data = []
    open(ARGS[2], "r") do file
        # reference: https://discourse.julialang.org/t/how-to-convert-dataframe-into-array-t-2/22487
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        data = CSV.read(file)[:, 2:end] |> Array{Float64, 2}
    end
    data
end
# upperTimeLowerLocation
const ul = let
    data = []
    open(ARGS[3], "r") do file
        # reference: https://discourse.julialang.org/t/how-to-convert-dataframe-into-array-t-2/22487
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        data = CSV.read(file)[:, 2:end] |> Array{Float64, 2}
    end
    data
end
# upperTimeUpperLocation
const uu = let
    data = []
    open(ARGS[4], "r") do file
        # reference: https://discourse.julialang.org/t/how-to-convert-dataframe-into-array-t-2/22487
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        # reference: https://qiita.com/Y0KUDA/items/3d3342ef08b28d5cda71
        data = CSV.read(file)[:, 2:end] |> Array{Float64, 2}
    end
    data
end
# get latitudes ad longitudes
const latitudes, longitudes, points = let
    data = []
    open(ARGS[1], "r") do file
        data = CSV.read(file)
    end
    latitudes = data[:, [1]] |> Array{Float64}
    longitudes = [ parse(Float64, element) for element in names(data)[2:end] |> Array{String} ]
    points = size(longitudes)[1]
    (latitudes, longitudes, points)
end

# lower time
const tl = parse(Float64, ARGS[5])
# upper time
const tu = parse(Float64, ARGS[6])
# the specific time needed to be calculated
const t = parse(Float64, ARGS[7])

const lowerLayer = (ul - ll) ./ (tu - tl) .* (t - tl) + ll
const upperLayer = (uu - lu) ./ (tu - tl) .* (t - tl) + lu

const lowestLatitude = latitudes[1]
const lowestLongitude = longitudes[1]
const xyLowerB = Boundary(;value=lowerLevel, f=(i, j, z) -> lowerLayer[i, j], type='D')
const xyUpperB = Boundary(;value=upperLevel, f=(i, j, z) -> upperLayer[i, j], type='D')
const yzLowerB = Boundary(;value=latitudes[1], f=(x, y, z) -> 0, type='N')
const yzUpperB = Boundary(;value=latitudes[end], f=(x, y, z) -> 0, type='N')
const xzLowerB = Boundary(;value=longitudes[1], f=(x, y, z) -> 0, type='N')
const xzUpperB = Boundary(;value=longitudes[end], f=(x, y, z) -> 0, type='N')

const dirName = "Interpolation"

if isdir("./$dirName")
    rm("./$dirName", recursive=true)
    mkdir("./$dirName")
else
    mkdir("./$dirName")
end
const results = @time solve3DEllipticPDE(; xyLowerB=xyLowerB, xyUpperB=xyUpperB, yzLowerB=yzLowerB, yzUpperB=yzUpperB, xzLowerB=xzLowerB, xzUpperB=xzUpperB, points=points)
const files = saveResults(; dirName=dirName, xs=LinRange(yzLowerB.value, yzUpperB.value, points), ys=LinRange(xzLowerB.value, xzUpperB.value, points), zs=LinRange(xyLowerB.value, xyUpperB.value, points), results=results)
# run(`python3 plotInterpolation.py $files`)
