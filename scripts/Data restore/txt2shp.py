import os
import shapefile
from shapely.geometry import LineString


# ----------------------------------------------------------------------
# Input and output paths
# ----------------------------------------------------------------------
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 data 文件夹路径（向上两级）
data_dir = os.path.join(current_dir, '..', '..', 'data')

# Directory containing decoded point coordinate text files
txt_file_path =os.path.join(data_dir, f"simplified_points_dp_5.0.txt")

# Output shapefile path
output_shp = os.path.join(data_dir, "DP_test.shp")


# ----------------------------------------------------------------------
# Prepare output directory
# ----------------------------------------------------------------------

# Create output directory if it does not exist
output_dir = os.path.dirname(output_shp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ----------------------------------------------------------------------
# Convert point sequences to polyline shapefile
# ----------------------------------------------------------------------

# Create a Shapefile writer for polyline geometries
with shapefile.Writer(output_shp, shapeType=shapefile.POLYLINE) as w:

    # Define attribute fields
    w.field("ID", "N")

    i = 0  # Line feature ID counter

    # Iterate over point files with incremental naming convention
    while True:
        txt_file = os.path.join(txt_file_path, f"{i}_point.txt")

        # Stop processing when the next file does not exist
        if not os.path.exists(txt_file):
            break

        # Read coordinate data from text file
        with open(txt_file, "r") as f:
            lines = f.readlines()

        # Parse coordinates (x, y) from each line
        coordinates = [
            tuple(map(float, line.strip().split(",")))
            for line in lines
        ]

        # Construct a polyline from the decoded point sequence
        line = LineString(coordinates)

        # Write polyline geometry to the shapefile
        w.line([list(line.coords)])

        # Write feature ID to the attribute table
        w.record(i + 1)

        print(f"Line feature {i + 1} successfully saved.")

        i += 1  # Proceed to the next file


print(f"All polyline features have been saved to {output_shp}.")
