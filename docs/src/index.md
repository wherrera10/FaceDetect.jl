# FaceDetect.jl

![Description](assets/childface.png)

OpenCV based face detection in Julia

## Examples

```julia
using FaceDetect
using OpenCV

# Load the built-in Haar cascade files if needed
FaceDetect.loadcascades()

# Read an image and detect faces: returns vector of bounding boxes for detected faces
img = OpenCV.imread("groupphoto.jpg", OpenCV.IMREAD_GRAYSCALE)
faces = FaceDetect.facedetect(img; biggest = true, min_neighbors = 5)

println("Detected $(length(faces)) face(s)")
for (x, y, w, h) in faces
    println("Face: x=$x, y=$y, width=$w, height=$h")
end


# Another example: this returns (equalized_image, detected_faces)
img, faces = FaceDetect.facedetect("groupphoto.jpg"; biggest = true)

println("Detected $(length(faces)) face(s)")
for (x, y, w, h) in faces
    println("Face: x=$x, y=$y, width=$w, height=$h")
end


# And if you want a visual example, with use of OpenCV display functions:

using FaceDetect
using OpenCV

loadcascades() # Call before using facedetect, to set up the CASCADES dict. Downloads data if missing.

# Manual image handling: facedetect's size filtering assumes a single-channel image,
# so detect on grayscale and draw onto a new color copy for display/output.

img_gray = OpenCV.imread("groupphoto.jpg", OpenCV.IMREAD_GRAYSCALE)
img = OpenCV.imread("groupphoto.jpg")
faces = FaceDetect.facedetect(img_gray; biggest = false, min_neighbors = 5)

for (x, y, w, h) in faces
    p1 = OpenCV.Point{Int32}(Int32(x), Int32(y))
    p2 = OpenCV.Point{Int32}(Int32(x + w), Int32(y + h))
    OpenCV.rectangle(img, p1, p2, (0, 255, 255); thickness = 2)
end

OpenCV.imwrite("faces_detected.jpg", img)
OpenCV.imshow("Detected Faces", img)
OpenCV.waitKey(0)
OpenCV.destroyAllWindows()

```


## Installation
```julia
using Pkg
Pkg.add("FaceDetect")
```

## Functions Reference


```@index
```

```@autodocs
Modules = [FaceDetect]
```
