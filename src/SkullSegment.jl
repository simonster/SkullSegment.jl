module SkullSegment
using NIfTI
include(joinpath(Pkg.dir("NIfTI"), "examples", "register.jl"))

immutable XYZ
    x::Int
    y::Int
    z::Int
end

Base.setindex!{T<:Real}(X::Array{T}, val::T, x::XYZ) = (X[x.x, x.y, x.z] = val)
Base.getindex{T<:Real}(X::Array{T}, x::XYZ) = X[x.x, x.y, x.z]

function process!(processed, to_process, v::XYZ)
    if !processed[v]
        processed[v] = true
        push!(to_process, v)
    end
end

function floodfill(arr, seed::(Int, Int, Int), lb, ub)
    processed = zeros(Bool, size(arr))
    out = zeros(Bool, size(arr))
    to_process = [XYZ(seed[1], seed[2], seed[3])]
    processed[seed[1], seed[2], seed[3]] = 1
    while !isempty(to_process)
        cur = pop!(to_process)
        out[cur] = true
        if lb <= arr[cur] <= ub
            cur.x > 1 && process!(processed, to_process, XYZ(cur.x-1, cur.y, cur.z))
            cur.x < size(arr, 1) && process!(processed, to_process, XYZ(cur.x+1, cur.y, cur.z))
            cur.y > 1 && process!(processed, to_process, XYZ(cur.x, cur.y-1, cur.z))
            cur.y < size(arr, 2) && process!(processed, to_process, XYZ(cur.x, cur.y+1, cur.z))
            cur.z > 1 && process!(processed, to_process, XYZ(cur.x, cur.y, cur.z-1))
            cur.z < size(arr, 3) && process!(processed, to_process, XYZ(cur.x, cur.y, cur.z+1))
        end
    end
    out
end

# Add a suffix to the part of a filename before the .
function addsuffix(path::String, suffix::String)
    dir = dirname(path)
    base = basename(path)
    dotindex = rsearch(base, '.')
    joinpath(dir, base[1:dotindex-1]*suffix*base[dotindex:end])
end

# Average several anatomical volumes
function avg{T<:String}(paths::Vector{T}, out::String)
	!isempty(paths) || error("no volume specified")
	ni = niread(paths[1], mmap=true)
	affine = getaffine(ni.header)
	niraw = isa(ni.raw, Array{Float32}) ? copy(ni.raw) : float32(ni.raw)
	for i = 2:length(paths)
		ni2 = niread(paths[i], mmap=true)
		size(ni2) == size(ni) || error("volume sizes do not match")
		affine == getaffine(ni2.header) || warn("volumes have different affine matrices")
		broadcast!(+, niraw, ni2.raw)
	end
	scale!(niraw, 1/length(paths))
	niwrite(out, NIVolume(ni.header, niraw))
end

# Normalize T1 scan using N3
function n3normalize(path::String)
    nu = addsuffix(path, "nu")
    wm110 = addsuffix(path, "wm110")

    # Pretend the dimensions are 1 mm in each direction because
    # FreeSurfer is stupid.
    # XXX This probably doesn't work if the slices aren't iso
    nutemp = tempname()*".nii"
    ni = niread(path, mmap=true)
    origheader = deepcopy(ni.header)
    ni.header.pixdim[2:4] = float32(1)
    ni.header.xyzt_units = int8(2) # NIfTI_UNITS_MM
    ni.header.srow_x /= origheader.pixdim[2]
    ni.header.srow_y /= origheader.pixdim[3]
    ni.header.srow_z /= origheader.pixdim[4]
    niwrite(nutemp, ni)

    # Correct for inhomogeneity due to coils
    # Optimized parameters from Zheng, W., Chee, M. W. L., &
    # Zagorodnov, V. (2009). Improvement of brain segmentation accuracy
    # by optimizing non-uniformity correction using N3. NeuroImage,
    # 48(1), 73–83. doi:10.1016/j.neuroimage.2009.06.039
    try
        run(`mri_nu_correct.mni --i $nutemp --o $nu --proto-iters 1000 --distance 30`)
    finally
        rm(nutemp)
    end

    # Normalize white matter to 110
    run(`mri_normalize -monkey $nu $wm110 -v -mprage -nosnr`)

    # Fix the dimensions
    for vol in (nu, wm110)
        ni = niread(vol, mmap=true)
        ni.header.pixdim[2:4] = origheader.pixdim[2:4]
        ni.header.xyzt_units = origheader.xyzt_units
        ni.header.srow_x *= origheader.pixdim[2]
        ni.header.srow_y *= origheader.pixdim[3]
        ni.header.srow_z *= origheader.pixdim[4]
        f = open(vol, "r+")
        write(f, ni.header)
        close(f)
    end

    (nu, wm110)
end

# Generate brain mask for anatomical
function extractbrain(path::String; threshold=0.4, flood=20)
    norm = addsuffix(path, "wm110")
    isfile(norm) || n3normalize(path)

    masked = addsuffix(path, "masked")
    run(`mri_gcut $norm $masked -110 -T $threshold`)

    mask = addsuffix(path, "mask")
    ni = niread(masked)
    if flood != 0
        ni.raw[floodfill(ni.raw, (1, 1, 1), 0, flood)] = 0
        niwrite(masked, ni)
    end

    maskvol = NIVolume(ni.header, uint8(ni.raw .!= 0))
    maskvol = register(niread(path, mmap=true), maskvol)
    niwrite(mask, maskvol)
    mask
end

# View brain
# function viewbrain(vol::NIVolume)
#     f = figure()
#     @manipulate for z = 1:size(vol, 3); withfig(f) do
#         imshow(vol.raw[:, :, z].', cmap="gray", origin="upper")
#     end end
# end
# viewbrain(vol::String) = viewbrain(niread(vol))
end