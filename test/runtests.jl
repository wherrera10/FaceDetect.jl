using Test
using FaceDetect
using OpenCV
using Statistics

FaceDetect.DEBUG_PRINTLN[] = false # suppress stdout error messages from fatal()

@testset "grayarray" begin
    a = rand(UInt8, 50, 60)
    @test size(FaceDetect.grayarray(a)) == (50, 60)

    b = reshape(a, 1, 50, 60)
    g = FaceDetect.grayarray(b)
    @test size(g) == (50, 60)
    @test g == a

    @test_throws ErrorException FaceDetect.grayarray(rand(UInt8, 3, 50, 60))
end

@testset "croprect" begin
    im = reshape(collect(1:100), 1, 10, 10)
    rect = (2, 3, 4, 5) # x, y, w, h (0-based)
    cropped = FaceDetect.croprect(im, rect)
    @test size(cropped) == (1, 4, 5)

    cropped2 = FaceDetect.croprect(im, rect, 1)
    @test size(cropped2) == (1, 2, 3)
end

@testset "shavemargin" begin
    a2 = rand(20, 30)
    @test size(FaceDetect.shavemargin(a2, 3)) == (14, 24)
    @test size(FaceDetect.shavemargin(a2, 0)) == (20, 30)

    a3 = rand(1, 20, 30)
    @test size(FaceDetect.shavemargin(a3, 2)) == (1, 16, 26)
end

@testset "normalizerect" begin
    im = rand(UInt8, 1, 200, 200)
    rect = (40, 60, 80, 100)

    n = FaceDetect.normalizerect(im, rect; equalize = false, same_aspect = false)
    @test size(n) == (FaceDetect.NORM_SIZE, FaceDetect.NORM_SIZE)

    n2 = FaceDetect.normalizerect(im, rect; equalize = false, same_aspect = true)
    @test maximum(size(n2)) == FaceDetect.NORM_SIZE
end

@testset "rankfaces" begin
    im = rand(UInt8, 1, 300, 300)
    rects = [(50, 50, 80, 80), (100, 100, 40, 40), (20, 20, 120, 120)]

    scores, best = FaceDetect.rankfaces(im, rects)
    @test length(scores) == 3
    @test best isa Integer
    @test 1 ≤ best ≤ 3

    ranks = [s["RANK"] for s in scores]
    @test sort(ranks) == 0:2

    scores∅, best∅ = FaceDetect.rankfaces(im, [])
    @test isempty(scores∅)
    @test best∅ === nothing
end

@testset "mssim_norm" begin
    X = rand(Float32, 64, 64)
    Y = copy(X)

    s = FaceDetect.mssim_norm(X, Y)
    @test s ≈ 1.0 atol = 1.0e-4

    Z = rand(Float32, 64, 64)
    s2 = FaceDetect.mssim_norm(X, Z)
    @test -1.0 ≤ s2 ≤ 1.0

    @test_throws AssertionError FaceDetect.mssim_norm(X, rand(Float32, 1, 32, 32))
end

@testset "parseargs" begin
    opts = FaceDetect.parseargs(["--biggest", "--best", "-c", "img.jpg"])
    @test opts[:biggest]
    @test opts[:best]
    @test opts[:center]
    @test opts[:file] == "img.jpg"
    @test opts[:min_neighbors] == 5
    @test opts[:search_threshold] == 30

    opts2 = FaceDetect.parseargs([
        "--min-neighbors",
        "8",
        "--search-threshold",
        "45",
        "photo.png",
    ])
    @test opts2[:min_neighbors] == 8
    @test opts2[:search_threshold] == 45
    @test opts2[:file] == "photo.png"

    @test_throws ErrorException FaceDetect.parseargs(["--biggest"])
    @test_throws ErrorException FaceDetect.parseargs(["--foo", "a.jpg"])
    @test_throws ErrorException FaceDetect.parseargs(["a.jpg", "b.jpg"])
end

@testset "detection (requires cascades)" begin
    loadcascades()

    @test haskey(FaceDetect.CASCADES, "HAAR_FRONTALFACE_ALT2")
    @test !OpenCV.empty(FaceDetect.CASCADES["HAAR_FRONTALFACE_ALT2"])

    blank = zeros(UInt8, 1, 200, 200)
    rects = facedetect(blank; min_neighbors = 5)
    @test isempty(rects)

    _, features = facedetect("testfile8.jpg")
    @test length(features) == 8
    @test (1066, 363, 94, 94) ∈ features
end

true
