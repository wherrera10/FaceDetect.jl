const DATA_DIR = joinpath(@__DIR__, "opencv-data")
const PROFILES = Dict(
    "HAAR_FRONTALFACE_ALT2" => "haarcascades/haarcascade_frontalface_alt2.xml",
)
const CASCADE_BASE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/"

"""
    loadcascades(data_dir=DATA_DIR)

Load the Haar cascade classifiers listed in `PROFILES` from `data_dir`
into the global `CASCADES` dictionary.
"""
function loadcascades(data_dir = DATA_DIR)
    empty!(CASCADES)
    for (name, relative_path) in PROFILES
        path = joinpath(data_dir, relative_path)
        if !isfile(path)
            downloadcascadexml(relative_path, path)
        end
        cascade = OpenCV.CascadeClassifier(path)
        if OpenCV.empty(cascade)
            fatal("cannot load $name from $path")
        end
        CASCADES[name] = cascade
    end
    return nothing
end

"""
    downloadcascadexml(relative_path, dest_path)

Download a Haar cascade XML file from the official OpenCV GitHub
repository into `dest_path` if it is not already present locally.
"""
function downloadcascadexml(relative_path, dest_path)
    mkpath(dirname(dest_path))
    url = CASCADE_BASE_URL * relative_path
    try
        Downloads.download(url, dest_path)
    catch e
        fatal("failed to download cascade from $url: $e")
    end
    return nothing
end
