import shapefile
from shapely.geometry import LineString, Point
import os
import shutil
from shapely.ops import linemerge
import sys


# ----------------------------------------------------------------------
# Line splitting and merging utilities
# ----------------------------------------------------------------------

def split_line(line, distance):
    """
    Split a polyline into segments based on a fixed accumulated distance.

    The line is traversed sequentially, and split points are inserted
    whenever the accumulated length reaches the specified distance.
    This method considers the cumulative length across multiple segments.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Input polyline geometry.
    distance : float
        Target distance interval for splitting.

    Returns
    -------
    list of tuple
        A list of tuples (Point, split_flag), where split_flag indicates
        whether the point is a splitting point (1) or not (0).
    """

    coords = list(line.coords)
    accumulated_distance = 0.0
    last_point = Point(coords[0])

    # Initialize split points with the starting point (split_p = 0)
    split_points = [(last_point, 0)]

    for i in range(1, len(coords)):
        current_point = Point(coords[i])
        segment = LineString([last_point, current_point])
        segment_length = segment.length

        # Insert split points when accumulated distance exceeds threshold
        while accumulated_distance + segment_length >= distance:
            remaining_distance = distance - accumulated_distance
            split_point = segment.interpolate(
                remaining_distance / segment_length,
                normalized=True
            )

            split_points.append((split_point, 1))

            # Reset accumulation after inserting a split point
            last_point = split_point
            accumulated_distance = 0.0
            segment = LineString([last_point, current_point])
            segment_length = segment.length

        accumulated_distance += segment_length
        last_point = current_point

        # Append the current vertex (not a split point)
        split_points.append((current_point, 0))

    return split_points


# ----------------------------------------------------------------------
# Input parameters and file paths
# ----------------------------------------------------------------------
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 data 文件夹路径（向上两级）
data_dir = os.path.join(current_dir, '..', '..', 'data')

# 转为绝对路径，避免路径问题
data_dir = os.path.abspath(data_dir)

input_shp = os.path.join(data_dir, 'Line_Island.shp')
output_shp = os.path.join(data_dir, 'Seg_Line_Island.shp')

# Fixed distance interval for line splitting
distance = 25000


# ----------------------------------------------------------------------
# Shapefile processing
# ----------------------------------------------------------------------

# Read input shapefile
sf = shapefile.Reader(input_shp)

# Create output shapefile writer
w = shapefile.Writer(output_shp)

# Copy original fields and append new attributes
w.fields = sf.fields[1:]
w.field('split_p', 'N', 1)
w.field('split_line', 'N', 6)

# Copy projection file (.prj) if it exists
prj_file = input_shp.replace('.shp', '.prj')
if os.path.exists(prj_file):
    shutil.copy(
        prj_file,
        output_shp.replace('.shp', '.prj')
    )

# Process each feature in the shapefile
for shape_record in sf.shapeRecords():

    line = LineString(shape_record.shape.points)

    # Split the polyline into points with split indicators
    split_points = split_line(line, distance)

    # Initialize split_line counter
    split_line_counter = 0
    split_line_dict = {}

    # Generate new line segments and group them by split_line attribute
    for i in range(len(split_points) - 1):
        new_line = LineString(
            [split_points[i][0], split_points[i + 1][0]]
        )

        # Extract split indicator
        split_p = split_points[i + 1][1]

        # Initialize list for current split_line if necessary
        if split_line_counter not in split_line_dict:
            split_line_dict[split_line_counter] = []

        # Append segment to the corresponding split_line group
        split_line_dict[split_line_counter].append(new_line)

        # Increment split_line counter when a split point is encountered
        split_line_counter += split_p

    # Merge segments belonging to the same split_line group
    for split_line_value, line_segments in split_line_dict.items():

        merged_line = linemerge(line_segments)

        # Ensure the merged geometry is a single LineString
        if isinstance(merged_line, LineString):

            # Write merged geometry
            w.line([list(merged_line.coords)])

            # Copy original attributes and append split information
            w.record(
                *shape_record.record,
                0,
                split_line_value
            )

        else:
            # Terminate if merging results in multiple geometries
            print(
                f"Merged result contains multiple line geometries, "
                f"record attributes: {shape_record.record}"
            )

# Close the shapefile writer
w.close()
