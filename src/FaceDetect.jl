""" Julia face detection using OpenCV.jl. """
module FaceDetect

using OpenCV
using Downloads, Printf, Statistics

export loadcascades, facedetect

const NORM_SIZE = 100
const NORM_MARGIN = 10
const CASCADES = Dict{String, OpenCV.CascadeClassifier}()

const DEBUG_PRINTLN = Ref(true)

include("cascades.jl")
include("appfuncs.jl")

"""
    fatal(msg)

Print an error message `msg` to stderr and throw an exception with the message."""
function fatal(msg)
    DEBUG_PRINTLN[] && println(stderr, "Error in FaceDetect.jl: $msg")
    throw(ErrorException(msg))
end

"""
    grayarray(im)

OpenCV.jl represents a grayscale Mat with a third, singleton channel dimension.
Drop the singleton and return a 2-dimensional matrix for numerical operations.
"""
function grayarray(im)
    a = Array(im)
    if ndims(a) == 3 && size(a, 1) == 1
        return dropdims(a; dims = 1)
    elseif ndims(a) == 2
        return a
    else
        fatal("Needed 2D or singleton 3D grayscale image, got size $(size(a))")
    end
end

"""
    croprect(im, rect, shave=0)

Crop `im` to the rectangle `rect = (x, y, w, h)`, optionally shaving
`shave` pixels off each edge. Converts from OpenCV's zero-based
coordinates to Julia's one-based array indexing. Returns partial copy of the image.
"""
function croprect(im, rect, shave = 0)
    x, y, w, h = rect
    x1 = x + shave + 1
    y1 = y + shave + 1
    x2 = x + w - shave
    y2 = y + h - shave
    return im[:, x1:x2, y1:y2]
end

"""
    shavemargin(im, margin)

Remove `margin` pixels from each edge of `im`, supporting both 2-D
and channel-first 3-D arrays.
"""
function shavemargin(im, margin)
    if margin == 0
        return im
    end
    if ndims(im) == 3
        return im[:, margin + 1:end - margin, margin + 1:end - margin]
    else
        return im[margin + 1:end - margin, margin + 1:end - margin]
    end
end

"""
    normalizerect(im, rect; equalize=true, same_aspect=false)

Crop and resize the face region `rect` from `im` to a normalized
size. The OpenCV portion remains an OpenCV Mat. Once normalization is
complete, the single-channel image is converted to a 2-D Julia array.
"""
function normalizerect(im, rect; equalize = true, same_aspect = false)
    roi = croprect(im, rect)
    if equalize
        roi = OpenCV.equalizeHist(roi)
    end

    side = NORM_SIZE + NORM_MARGIN* 2
    _, _, w, h = rect

    if same_aspect
        scale = side / max(w, h)
        new_width = max(1, round(Int, w * scale))
        new_height = max(1, round(Int, h * scale))
        dsize = OpenCV.Size{Int32}(Int32(new_width), Int32(new_height))
    else
        dsize = OpenCV.Size{Int32}(Int32(side), Int32(side))
    end

    roi = OpenCV.resize(roi, dsize; interpolation = OpenCV.INTER_CUBIC)
    roi = shavemargin(roi, NORM_MARGIN)

    return grayarray(roi)
end

"""
    rankfaces(im, rects)

Score and rank candidate face rectangles `rects` detected in `im`,
returning the per-rectangle scores and the index of the best match.
"""
function rankfaces(im, rects)
    scores = Dict{String, Any}[]
    isempty(rects) && return scores, nothing

    # OpenCV.jl image dimensions are: (channels, width, height)
    width = size(im, 2)
    height = size(im, 3)

    for rect in rects
        x, y, w, h = rect

        # normalizerect returns a 2-D grayscale array.
        roi_n = normalizerect(im, rect; equalize = false, same_aspect = true)
        # OpenCV.Laplacian requires a channel-first 3-D array, not 2-D
        roi_n3 = reshape(roi_n, 1, size(roi_n)...)
        # Float32 depth avoids clipping negative curvature and variance measures edge energy
        roi_l = OpenCV.Laplacian(roi_n3, OpenCV.CV_32F)
        e = var(roi_l)

        dx = width / 2 - x + w / 2
        dy = height / 2 - y + h / 2
        d = sqrt(dx^2 + dy^2) / (max(width, height) / 2)
        s = (w + h) / 2

        push!(scores, Dict("s" => s, "e" => e, "d" => d))
    end

    smax = maximum(s["s"] for s in scores)
    emax = maximum(s["e"] for s in scores)

    for score in scores
        s = score["s"]
        e = score["e"]
        d = score["d"]

        sN = s / smax
        eN = e / emax

        score["sN"] = sN
        score["eN"] = eN
        score["f"] = eN * 0.7 + (1 - d) * 0.1 + sN * 0.2
    end

    ranks = sortperm(1:length(scores); by = i -> scores[i]["f"], rev = true)

    for (rank, index) in enumerate(ranks)
        scores[index]["RANK"] = rank - 1
    end

    return scores, ranks[1]
end

"""
    mssim_norm(X, Y; K1=0.01, K2=0.03, win_size=11, sigma=1.5)

Compute the mean structural similarity (MSSIM) between two images.
`X` and `Y` are floating point Matrix type arrays holding the image data.
"""
function mssim_norm(X, Y; K1 = 0.01, K2 = 0.03, win_size = 11, sigma = 1.5)
    # Ensure the numerical inputs are ordinary 2-D arrays.
    X = ndims(X) == 3 ? grayarray(X) : Array(X)
    Y = ndims(Y) == 3 ? grayarray(Y) : Array(Y)

    @assert ndims(X) == 2
    @assert ndims(Y) == 2
    @assert size(X) == size(Y)

    C1 = K1^2
    C2 = K2^2
    cov_norm = win_size^2

    kernel_size = OpenCV.Size{Int32}(Int32(win_size), Int32(win_size))

    # OpenCV.jl requires a 3-D array (channel dim first) for its filter ops.
    to3d(a) = reshape(a, 1, size(a)...)

    ux = OpenCV.GaussianBlur(to3d(X), kernel_size, sigma)
    uy = OpenCV.GaussianBlur(to3d(Y), kernel_size, sigma)
    uxx = OpenCV.GaussianBlur(to3d(X .* X), kernel_size, sigma)
    uyy = OpenCV.GaussianBlur(to3d(Y .* Y), kernel_size, sigma)
    uxy = OpenCV.GaussianBlur(to3d(X .* Y), kernel_size, sigma)

    ux = dropdims(Array(ux); dims = 1)
    uy = dropdims(Array(uy); dims = 1)
    uxx = dropdims(Array(uxx); dims = 1)
    uyy = dropdims(Array(uyy); dims = 1)
    uxy = dropdims(Array(uxy); dims = 1)

    vx = cov_norm .* (uxx .- ux .* ux)
    vy = cov_norm .* (uyy .- uy .* uy)
    vxy = cov_norm .* (uxy .- ux .* uy)

    A1 = 2 .* ux .* uy .+ C1
    A2 = 2 .* vxy .+ C2
    B1 = ux .* ux .+ uy .* uy .+ C1
    B2 = vx .+ vy .+ C2

    D = B1 .* B2
    S = (A1 .* A2) ./ D

    margin = (win_size - 1) ÷ 2

    return mean(shavemargin(S, margin))
end

"""
    facedetect(im::AbstractArray; biggest=false, scalefactor=1.1, min_neighbors=5, min_frac=1/20, max_frac=1/2)

Detect faces in the grayscale image `im` using the loaded Haar
cascade, optionally restricting the search to the biggest face.

`min_neighbors` controls how many overlapping candidate rectangles must
agree before a detection is accepted; raising it (from the OpenCV
default of ~3-4) trades recall for precision and is the most effective
lever for suppressing false-positive detections without new training
data. `scalefactor` and the `min_frac`/`max_frac` size bounds can be
tightened similarly to prune spurious small/large matches.
"""
function facedetect(
    im::AbstractArray;
    biggest = false,
    scalefactor = 1.1,
    min_neighbors = 5,
    min_frac = 1 / 20,
    max_frac = 1 / 2,
)
    # OpenCV.jl call below returns (channels x and y, width, height)
    # width = size(im, 2)
    # height = size(im, 3)

    side = sqrt(length(im))
    minlen = max(1, floor(Int, side * min_frac))
    maxlen = max(minlen, floor(Int, side * max_frac))

    # OpenCV: CASCADE_DO_CANNY_PRUNING = 1, CASCADE_FIND_BIGGEST_OBJECT = 4
    flags = Int32(1)
    if biggest
        flags |= Int32(4)
    end

    if !haskey(CASCADES, "HAAR_FRONTALFACE_ALT2")
        fatal("face cascade has not been loaded")
    end

    cascade = CASCADES["HAAR_FRONTALFACE_ALT2"]

    features = OpenCV.detectMultiScale(
        cascade,
        im;
        scaleFactor = Float64(scalefactor),
        minNeighbors = Int32(min_neighbors),
        flags = flags,
        minSize = OpenCV.Size{Int32}(Int32(minlen), Int32(minlen)),
        maxSize = OpenCV.Size{Int32}(Int32(maxlen), Int32(maxlen)),
    )

    return [(Int(r.x), Int(r.y), Int(r.width), Int(r.height)) for r in features]
end

"""
    facedetect(path::AbstractString; biggest=false)

Load the image at `path`, equalize its histogram, and run `facedetect` on it.
"""
function facedetect(path::AbstractString; biggest = false, min_neighbors = 5)
    im = OpenCV.imread(path, OpenCV.IMREAD_GRAYSCALE)
    if isempty(im.data)
        fatal("cannot load input image $path")
    end
    im = OpenCV.equalizeHist(im)
    features = facedetect(im; biggest = biggest, min_neighbors = min_neighbors)
    return im, features
end

"""
    pairwisesimilarity(im, features, template; mssim_args...)

Compute the MSSIM score between `template` and each detected face
rectangle in `features`, after normalizing each face region.
"""
function pairwisesimilarity(im, features, template; mssim_args...)
    template = Float32.(template) ./ 255
    scores = Float64[]

    for rect in features
        roi = normalizerect(im, rect)
        roi = Float32.(roi) ./ 255
        score = mssim_norm(roi, template; mssim_args...)
        push!(scores, score)
    end

    return scores
end

"""
    drawresults(input_path, output_path, features, scores, best, debug)

Draw bounding boxes (and, in debug mode, score metrics) for the
detected `features` onto the image at `input_path`, then write the
result to `output_path`.
"""
function drawresults(input_path, output_path, features, scores, best, debug)
    im = OpenCV.imread(input_path)
    if isempty(im.data)
        fatal("cannot load input image $input_path")
    end

    for i in eachindex(features)
        if best !== nothing && i != best && !debug
            continue
        end

        x, y, w, h = features[i]
        fg = (255, 255, 255)
        if best !== nothing && i == best
            fg = (0, 255, 255)
        end

        p1 = OpenCV.Point{Int32}(Int32(x), Int32(y))
        p2 = OpenCV.Point{Int32}(Int32(x + w), Int32(y + h))

        OpenCV.rectangle(im, p1, p2, (0, 0, 0); thickness = Int64(4))
        OpenCV.rectangle(im, p1, p2, fg; thickness = Int64(2))

        if debug
            text_y = y + h + 20
            for (key, value) in scores[i]
                text = "$key: $value"
                OpenCV.putText(
                    im,
                    text,
                    OpenCV.Point{Int32}(Int32(x), Int32(text_y)),
                    OpenCV.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    fg;
                    thickness = Int64(1),
                    lineType = OpenCV.LINE_AA,
                )
                text_y += 15
            end
        end
    end

    if !OpenCV.imwrite(output_path, im)
        fatal("cannot write output image $output_path")
    end

    return nothing
end


end # module FaceDetect
