import os
import shutil
import shapefile
from shapely.geometry import LineString, MultiLineString


def simplify_shapefile_dp(input_shp, output_shp, tolerance):
    """
    读取 Shapefile，使用 DP 算法压缩几何特征，输出新的 Shapefile，并计算压缩率。
    """
    # 1. 复制投影文件 (.prj)
    prj_file = input_shp.replace('.shp', '.prj')
    if os.path.exists(prj_file):
        shutil.copy(prj_file, output_shp.replace('.shp', '.prj'))

    output_dir = os.path.dirname(output_shp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化点数计数器
    total_input_points = 0
    total_output_points = 0
    feature_count = 0

    # 2. 读写 Shapefile
    with shapefile.Reader(input_shp) as sf:
        with shapefile.Writer(output_shp, shapeType=sf.shapeType) as w:
            w.fields = sf.fields[1:]

            for shape_record in sf.shapeRecords():
                shape = shape_record.shape
                points = shape.points
                parts = shape.parts

                if not points:
                    continue

                feature_count += 1
                # 累加当前要素的原始点数
                total_input_points += len(points)

                simplified_parts = []
                part_indices = list(parts) + [len(points)]

                # 基于 parts 进行分环处理
                for i in range(len(part_indices) - 1):
                    start_idx = part_indices[i]
                    end_idx = part_indices[i + 1]
                    ring_points = points[start_idx:end_idx]

                    if len(ring_points) >= 2:
                        line = LineString(ring_points)

                        # 执行 DP 简化
                        simplified_geom = line.simplify(tolerance, preserve_topology=True)

                        # 收集简化后的坐标并统计输出点数
                        if isinstance(simplified_geom, LineString):
                            coords = list(simplified_geom.coords)
                            simplified_parts.append(coords)
                            total_output_points += len(coords)

                        elif isinstance(simplified_geom, MultiLineString):
                            for geom in simplified_geom.geoms:
                                coords = list(geom.coords)
                                simplified_parts.append(coords)
                                total_output_points += len(coords)

                # 写入简化后的几何体和属性
                if simplified_parts:
                    if sf.shapeType == shapefile.POLYGON:
                        w.poly(simplified_parts)
                    elif sf.shapeType == shapefile.POLYLINE:
                        w.line(simplified_parts)

                    w.record(*shape_record.record)

    # 3. 计算并输出实验指标
    print(f"--- 压缩处理完成 ---")
    print(f"处理要素数量: {feature_count} 个")
    print(f"输入总点数 (N_orig): {total_input_points}")
    print(f"输出总点数 (N_simp): {total_output_points}")

    if total_output_points > 0:
        compression_ratio = total_input_points / total_output_points
        print(f"压缩率 (CR): {compression_ratio:.2f}x  (即 1:{compression_ratio:.2f})")
        print(f"数据量减少: {(1 - total_output_points / total_input_points) * 100:.2f}%")
    else:
        print("警告：输出点数为 0，可能阈值设置过大导致所有几何体被过度简化甚至消失。")

    print(f"简化后的 Shapefile 已保存至: {output_shp}")

    return total_input_points, total_output_points


# ==========================================
# 测试运行
# ==========================================
if __name__ == "__main__":
    # 替换为你的实际路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', '..', 'data')

    in_shp = os.path.join(data_dir, "Line_Island_part_DP.shp")
    out_shp = os.path.join(data_dir, "DP_test_220.shp")

    # 设置 DP 算法的距离阈值 (单位与你的投影坐标系一致，如米)
    dp_tolerance = 220.0

    simplify_shapefile_dp(in_shp, out_shp, dp_tolerance)