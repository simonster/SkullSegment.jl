module SkullSegment
using NIfTI, Compat
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

function floodfill(arr, seed::Tuple{Int, Int, Int}, lb, ub)
    lb <= arr[seed[1], seed[2], seed[3]] <= ub || warn("seed voxel not in range")
    processed = zeros(Bool, size(arr))
    out = zeros(Bool, size(arr))
    to_process = [XYZ(seed[1], seed[2], seed[3])]
    processed[seed[1], seed[2], seed[3]] = 1
    while !isempty(to_process)
        cur = pop!(to_process)
        if lb <= arr[cur] <= ub
            out[cur] = true
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
    joinpath(dir, base[1:dotindex-1]*"_"*suffix*base[dotindex:end])
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

function conform_header(origheader)
    header = deepcopy(origheader)
    header.pixdim = (header.pixdim[1], 1f0, 1f0, 1f0, header.pixdim[5:end]...)
    header.xyzt_units = 0x02 # NIfTI_UNITS_MM
    header.srow_x = map(x->x/origheader.pixdim[2], header.srow_x)
    header.srow_y = map(x->x/origheader.pixdim[3], header.srow_y)
    header.srow_z = map(x->x/origheader.pixdim[4], header.srow_z)
    header
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
    origheader = ni.header
    ni.header = conform_header(ni.header)
    niwrite(nutemp, ni)

    # Correct for inhomogeneity due to coils
    # Optimized parameters from Zheng, W., Chee, M. W. L., &
    # Zagorodnov, V. (2009). Improvement of brain segmentation accuracy
    # by optimizing non-uniformity correction using N3. NeuroImage,
    # 48(1), 73–83. doi:10.1016/j.neuroimage.2009.06.039
    try
        println(`mri_nu_correct.mni --i $nutemp --o $nu --proto-iters 1000 --distance 30`)
        run(`mri_nu_correct.mni --i $nutemp --o $nu --proto-iters 1000 --distance 30`)
    finally
        rm(nutemp)
    end

    # Normalize white matter to 110
    run(`mri_normalize -monkey $nu $wm110 -v -mprage -nosnr`)

    # Fix the dimensions
    for vol in (nu, wm110)
        ni = niread(vol, mmap=true)
        ni.header.pixdim = (ni.header.pixdim[1], origheader.pixdim[2:4]..., ni.header.pixdim[5:end]...)
        ni.header.xyzt_units = origheader.xyzt_units
        ni.header.srow_x = map(x->x*origheader.pixdim[2], ni.header.srow_x)
        ni.header.srow_y = map(x->x*origheader.pixdim[3], ni.header.srow_y)
        ni.header.srow_z = map(x->x*origheader.pixdim[4], ni.header.srow_z)
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

    # Register back to the original nu
    maskvol = NIVolume(ni.header, 0xff*(ni.raw .!= 0))
    nu = niread(addsuffix(path, "nu"), mmap=true)
    maskvol = register(nu, maskvol)
    niwrite(mask, maskvol)
    mask
end

function postprocess_skull!(skull, nu, threshold)
    # Clean up cases where we get too much skin
    for k = 1:size(skull, 3), j = 1:size(skull, 2)
        for i = 1:size(skull, 1)
            if skull[i, j, k] != 0
                for i2 = i:size(skull, 1)
                    if nu[i2, j, k] >= threshold
                        skull[i2, j, k] = 0
                    else
                        break
                    end
                end
                break
            end
        end
        for i = size(skull, 1):-1:1
            if skull[i, j, k] != 0
                for i2 = i:-1:1
                    if nu[i2, j, k] >= threshold
                        skull[i2, j, k] = 0
                    else
                        break
                    end
                end
                break
            end
        end
    end
    skull
end

# Extract skull
function extractskull(path::String; lower_threshold=0, upper_threshold=0, erode_threshold=90)
    norm = addsuffix(path, "nu")
    isfile(norm) || n3normalize(path)
    mask = addsuffix(path, "mask")
    isfile(mask) || extractbrain(path)

    maskvol = niread(mask)
    center = size(maskvol, 2) >> 1
    slice = maskvol.raw[:, center, :]
    proj = vec(sum(slice, 1))
    bottom = findfirst(proj)-20
    top = findlast(proj)+20
    proj = vec(sum(slice, 3))
    left = findfirst(proj)-30
    right = findlast(proj)+30
    slice = maskvol.raw[left+div(right-left+1, 2), :, bottom+div(top-bottom+1, 2)]
    proj = vec(slice)
    front = findfirst(proj)-30
    back = findlast(proj)+50
    center = div(back-front+1, 2)

    vol = niread(norm)
    volsubset = vol.raw[left:right, front:back, bottom:top]
    fill!(vol.raw, 0)
    vol.raw[left:right, front:back, bottom:top] = volsubset
    cropped = addsuffix(path, "cropped")
    # origheader = vol.header
    # vol.header = conform_header(origheader)
    niwrite(cropped, vol)

    masksubset = maskvol.raw[left:right, front:back, bottom:top]
    fill!(maskvol.raw, 0)
    maskvol.raw[left:right, front:back, bottom:top] = masksubset
    cropped_mask = addsuffix(path, "cropped_mask")
    # maskvol.header = conform_header(maskvol.header)
    niwrite(cropped_mask, maskvol)

    skull = addsuffix(path, "skull")
    run(`skullfinder -i $cropped -o $skull -m $cropped_mask -v 2 -l $lower_threshold -u $upper_threshold --scalplabel 0 --skulllabel 255 --spacelabel 255 --brainlabel 0`)

    # skullfinder screws up the registration, so fix it
    skullvol = niread(skull)
    skullvol.header = vol.header
    postprocess_skull!(skullvol.raw, vol.raw, erode_threshold)

    niwrite(skull, skullvol)
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