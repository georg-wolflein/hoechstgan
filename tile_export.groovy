import qupath.lib.images.servers.LabeledImageServer

print "Getting image data..."
def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).split('_')[0]

def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles',name)
mkdirs(pathOutput)

double avg_pixel_size = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
print avg_pixel_size

double downsample = 1
print downsample

print imageData.getServerPath()

// // Create an ImageServer where the pixels are derived from annotations
// def labelServer = new LabeledImageServer.Builder(imageData)
//     .useDetections()
//     .useCells()
//     .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
//     .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
//     .addLabel('CD3', 1)      // Choose output labels (the order matters!)
//     .addUnclassifiedLabel(2)
//     // .lineThickness(2)          // Optionally export annotation boundaries with another label
//     // .setBoundaryLabel('Boundary*', 3) // Define annotation boundary label
//     .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
//     .build()

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .useDetections()
    .useCells()
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('CD3', 1)      // Choose output labels (the order matters!)
    .addUnclassifiedLabel(2)
    .lineThickness(1)          // Optionally export annotation boundaries with another label
    .setBoundaryLabel('Boundary*', 0, ColorTools.WHITE) // Define annotation boundary label
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    //.imageSubDir("hoechst")
    .imageExtension('.tif')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    //.labeledImageSubDir("cd3")
    .labeledImageExtension(".png")
    .tileSize(256)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print pathOutput