# Manual

This page summarizes the intended workflow for using `OpenCVFaceDetection.jl` in real projects.

## Installation

```julia
using Pkg
Pkg.add("OpenCVFaceDetection")
```

Then load the package and OpenCV support:

```julia
using OpenCVFaceDetection
using OpenCV
```

## Downloading the cascade data

The Haar cascade files are not bundled in the package by default. Call `loadcascades()` before running detection so the library downloads and caches the required OpenCV models.

```julia
loadcascades()
```

## Detecting faces from an image array

```julia
img = OpenCV.imread("groupphoto.jpg", OpenCV.IMREAD_GRAYSCALE)
faces = facedetect(img; biggest = true, min_neighbors = 5)
```

`faces` is returned as a vector of tuples of the form `(x, y, width, height)`, expressed in image-coordinate space.

## Detecting faces from a file path

```julia
img, faces = facedetect("groupphoto.jpg"; biggest = true)
```

This path-based variant equalizes the image histogram before calling `facedetect` and returns the normalized image together with the detected rectangles.

## Drawing detections on a color image

```julia
img_gray = OpenCV.imread("groupphoto.jpg", OpenCV.IMREAD_GRAYSCALE)
img = OpenCV.imread("groupphoto.jpg")
faces = facedetect(img_gray; biggest = false, min_neighbors = 5)

for (x, y, w, h) in faces
    p1 = OpenCV.Point{Int32}(Int32(x), Int32(y))
    p2 = OpenCV.Point{Int32}(Int32(x + w), Int32(y + h))
    OpenCV.rectangle(img, p1, p2, (0, 255, 255); thickness = 2)
end

OpenCV.imwrite("faces_detected.jpg", img)
```

## Command-line usage

The package ships with a command-line script in `bin/` for simple batch processing:

```bash
./bin/facedetect -o detectedfaces.png groupphoto.jpg
```

The `-o` option indicates the output filename. If omitted, the last positional argument is treated as the input file.

## API summary

The package exports the main entry points:

- `loadcascades()`
- `facedetect(...)`

More detailed function-level documentation is available in the [API Reference](@ref).
