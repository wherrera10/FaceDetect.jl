
"""
    printhelp()

Print the command-line usage message for the facedetect tool.
"""
function printhelp()
    println(
        """
Usage: facedetect [options] FILE

A simple face detector for batch processing.

Options:
    --biggest
        Extract only the biggest face.

    --best
        Extract only the best matching face.

    -c, --center
        Print only the center coordinates.

    --data-dir DIRECTORY
        OpenCV data files directory.

    -q, --query
        Query only.
        Exit 0 if a face is detected, 2 otherwise.

    -s, --search FILE
        Search for faces similar to the one supplied in FILE.

    --search-threshold PERCENT
        Face similarity threshold (default: 30%).

    --min-neighbors N
        Minimum number of overlapping rectangles required to retain detection (default: 5).

    -o, --output FILE
        Image output file.

    -d, --debug
        Add debugging metrics to the image output file.

    -h, --help
        Show this help.
""",
    )
end

"""
    parseargs(args)

Parse the facedetect command-line arguments into an options dictionary.
"""
function parseargs(args)
    options = Dict{Symbol, Any}(
        :biggest => false,
        :best => false,
        :center => false,
        :data_dir => DATA_DIR,
        :query => false,
        :search => nothing,
        :search_threshold => 30,
        :min_neighbors => 5,
        :output => nothing,
        :debug => false,
        :file => nothing,
    )

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--biggest"
            options[:biggest] = true
        elseif arg == "--best"
            options[:best] = true
        elseif arg == "-c" || arg == "--center"
            options[:center] = true
        elseif arg == "--data-dir"
            i += 1
            i > length(args) && fatal("--data-dir requires a directory")
            options[:data_dir] = args[i]
        elseif arg == "-q" || arg == "--query"
            options[:query] = true
        elseif arg == "-s" || arg == "--search"
            i += 1
            i > length(args) && fatal("--search requires a file")
            options[:search] = args[i]
        elseif arg == "--search-threshold"
            i += 1
            i > length(args) && fatal("--search-threshold requires a value")
            options[:search_threshold] = parse(Int, args[i])
        elseif arg == "--min-neighbors"
            i += 1
            i > length(args) && fatal("--min-neighbors requires a value")
            options[:min_neighbors] = parse(Int, args[i])
        elseif arg == "-o" || arg == "--output"
            i += 1
            i > length(args) && fatal("--output requires a file")
            options[:output] = args[i]
        elseif arg == "-d" || arg == "--debug"
            options[:debug] = true
        elseif arg == "-h" || arg == "--help"
            printhelp()
            exit(0)
        elseif startswith(arg, "-")
            fatal("unknown option $arg")
        else
            if options[:file] !== nothing
                fatal("only one input file may be specified")
            end
            options[:file] = arg
        end

        i += 1
    end

    options[:file] === nothing && fatal("no input file specified")

    return options
end

"""
    facedetect(args=ARGS)

Entry point for the facedetect command-line tool: parses options,
loads cascades, detects (and optionally searches for) faces, and
prints or draws the results.
"""
function facedetect(args::Vector{String} = ARGS)
    options = parseargs(args)
    loadcascades(options[:data_dir])

    im, features = facedetect(
        options[:file];
        biggest = options[:query] || options[:biggest],
        min_neighbors = options[:min_neighbors],
    )

    sim_scores = nothing

    if options[:search] !== nothing
        search_im, search_features = facedetect(options[:search]; biggest = true)
        isempty(search_features) && fatal("cannot detect face in template")

        sim_threshold = options[:search_threshold] / 100
        sim_template = normalizerect(search_im, search_features[1])
        all_scores = pairwisesimilarity(im, features, sim_template)

        sim_scores = Float64[]
        sim_features = NTuple{4, Int}[]

        for (i, score) in enumerate(all_scores)
            if score >= sim_threshold
                push!(sim_scores, score)
                push!(sim_features, features[i])
            end
        end

        features = sim_features
    end

    if options[:query]
        return isempty(features) ? 2 : 0
    end

    scores = Dict{String, Any}[]
    best = nothing

    if !isempty(features) &&
            (
                options[:debug] ||
                    options[:best] ||
                    options[:biggest] ||
                    sim_scores !== nothing
            )

        scores, best = rankfaces(im, features)

        if sim_scores !== nothing
            for i in eachindex(features)
                scores[i]["MSSIM"] = sim_scores[i]
            end
        end
    end

    if options[:output] !== nothing
        drawresults(
            options[:file],
            options[:output],
            features,
            scores,
            best,
            options[:debug],
        )
    end

    if (options[:best] || options[:biggest]) && best !== nothing
        features = [features[best]]
    end

    if options[:center]
        for (x, y, w, h) in features
            cx = round(Int, x + w / 2)
            cy = round(Int, y + h / 2)
            println("$cx $cy")
        end
    else
        for (x, y, w, h) in features
            println("$x $y $w $h")
        end
    end

    return 0
end
