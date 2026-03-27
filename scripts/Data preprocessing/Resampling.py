import shapefile
from menuinst.platforms.win_utils.knownfolders import folder_path
from shapely.geometry import LineString, Point
import os
import shutil


# ----------------------------------------------------------------------
# Line resampling utilities
# ----------------------------------------------------------------------

def resample_line(line, num_points):
    """
    Resample a polyline using equal-distance interpolation.

    This method generates a fixed number of points that are evenly
    distributed along the normalized length of the input line.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Input polyline geometry.
    num_points : int
        Target number of resampled points.

    Returns
    -------
    shapely.geometry.LineString
        Resampled polyline with a fixed number of points.
    """

    # Generate normalized distances along the line [0, 1]
    distances = [i / (num_points - 1) for i in range(num_points)]

    # Interpolate points at equal normalized distances
    resampled_points = [
        line.interpolate(distance, normalized=True)
        for distance in distances
    ]

    return LineString(resampled_points)


def resample_line2(line, num_points):
    """
    Resample a polyline while preserving original vertices.

    This method iteratively inserts new points at the midpoint of the
    longest segment until the target number of points is reached.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Input polyline geometry.
    num_points : int
        Target number of resampled points.

    Returns
    -------
    shapely.geometry.LineString
        Resampled polyline preserving original vertices.
    """

    # Ensure the target number of points is larger than the original
    if len(line.coords) >= num_points:
        raise ValueError(
            "num_points must be greater than the number of original points"
        )

    # Initialize with original coordinates
    points = list(line.coords)

    while len(points) < num_points:
        max_distance = 0
        max_index = 0

        # Identify the longest segment
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            distance = LineString([p1, p2]).length

            if distance > max_distance:
                max_distance = distance
                max_index = i

        # Insert a new point at the midpoint of the longest segment
        new_point = LineString(
            [points[max_index], points[max_index + 1]]
        ).interpolate(0.5, normalized=True)

        points.insert(max_index + 1, new_point.coords[0])

    return LineString(points)


# ----------------------------------------------------------------------
# Parameters and file paths
# ----------------------------------------------------------------------

# Target number of points for each resampled polyline
num_points = 101

# Counter for exported coordinate text files
txt_num = 1

# Output folder for point coordinate text files
folderpath = (
    "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\point_对比岛屿"
)

# Input and output shapefile paths
input_shp = (
    "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\分段_对比岛屿.shp"
)
output_resampled_shp = (
    "C:\\Users\\ZS-1PC\\Desktop\\数据与代码\\海洋面All Step Data\\岛群\\重采样_对比岛屿.shp"
)


# ----------------------------------------------------------------------
# Shapefile processing
# ----------------------------------------------------------------------

# Read input shapefile
sf = shapefile.Reader(input_shp)

# Create output shapefile writer
w = shapefile.Writer(output_resampled_shp)

# Copy attribute fields
w.fields = sf.fields[1:]

# Copy projection file (.prj) if it exists
prj_file = input_shp.replace(".shp", ".prj")
if os.path.exists(prj_file):
    shutil.copy(
        prj_file,
        output_resampled_shp.replace(".shp", ".prj")
    )

# Iterate through all shape records
for shape_record in sf.shapeRecords():

    # Convert geometry to LineString
    line = LineString(shape_record.shape.points)

    # Resample the polyline
    resampled_line = resample_line(line, num_points)

    # Write resampled geometry to the output shapefile
    w.line([list(resampled_line.coords)])

    # Copy original attributes
    w.record(*shape_record.record)

    # Export resampled coordinates to a text file
    txt_file_path = os.path.join(
        folderpath,
        f"{txt_num}_resampled_coordinates.txt"
    )

    with open(txt_file_path, "w") as f:
        for coord in resampled_line.coords:
            f.write(f"{coord[0]}, {coord[1]}\n")

    # Update file counter
    txt_num += 1

# Close the shapefile writer
w.close()
