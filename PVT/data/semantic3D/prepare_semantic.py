import argparse
import math
import os
from datetime import datetime
import h5py
import numpy as np
import plyfile
from matplotlib import cm

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #BASE_DIR = 'C:\\Users\\Andrew\\Desktop\\CS674\\FinalProject\\full_dataset\\train\\clean\\'
    default_data =os.path.join(BASE_DIR,'full_dataset\\train\\clean\\points')
    default_label =os.path.join(BASE_DIR,'full_dataset\\train\\clean\\labels')
    default_h5dir = os.path.join(BASE_DIR,'full_dataset\\train\\clean\\h5dir')
    print(default_data)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest='data_dir', default=default_data,
                        help=f'Path to Semantic3D data (default is {default_data})')
    parser.add_argument('-f', '--folder', dest='output_dir', default=default_label,
                        help=f'Folder to write labels (default is {default_label})')
    parser.add_argument('--max_num_points', '-m', help='Max point number of each sample', type=int, default=1024)
    parser.add_argument('--block_size', '-b', help='Block size', type=float, default=1.5)
    parser.add_argument('--grid_size', '-g', help='Grid size', type=float, default=0.03)
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')

    args = parser.parse_args()
    print(args)

    #prepare_label(data_dir=args.data_dir, output_dir=args.output_dir)

    #root = args.output_dir
    max_num_points = args.max_num_points

    batch_size = 256
    data = np.zeros((batch_size, max_num_points, 10))
    data_num = np.zeros(batch_size, dtype=np.int32)
    label = np.zeros(batch_size, dtype=np.int32)
    label_seg = np.zeros((batch_size, max_num_points), dtype=np.int32)
    indices_split_to_full = np.zeros((batch_size, max_num_points), dtype=np.int32)

    # Modified according to PointNet convensions.
    datasets = [dataset for dataset in os.listdir(args.data_dir)]

    for dataset_idx, dataset in enumerate(datasets):
        #print(dataset)
        dataset_marker = os.path.join(args.data_dir, dataset[:-4]+'.dataset')
        if os.path.exists(dataset_marker):
            print(f'{datetime.now()}-{args.data_dir}/{dataset} already processed, skipping')
            continue
        dataset_file = os.path.join(args.data_dir, dataset)
        print(f'{datetime.now()}-Loading {dataset}...')
        print(dataset_file)
        xyzirgb = np.load(dataset_file)
        xyzirgb[:, 0:3] -= np.amin(xyzirgb, axis=0)[0:3]

        label_file = os.path.join(args.output_dir, dataset)
        labels = np.load(label_file).astype(int).flatten()

        xyz, intensity_rgb = np.split(xyzirgb, [3], axis=-1)
        intensity, rgb =  np.split(intensity_rgb, [1], axis=-1)

        # print(f"name:{dataset}\nxyz:{xyz}\nintensity:{intensity}\nrgb:{rgb}")

        xyz_min = np.amin(xyz, axis=0, keepdims=True)
        xyz_max = np.amax(xyz, axis=0, keepdims=True)
        xyz_center = (xyz_min + xyz_max) / 2
        xyz_center[0][-1] = xyz_min[0][-1]
        # Remark: Don't do global alignment.
        # xyz = xyz - xyz_center

        #normalizing intensity so between [0:1]
        intensity_min = np.amin(intensity, axis=0)
        intensity_max = np.amax(intensity, axis=0)
        intensity = (intensity - intensity_min) / (intensity_max - intensity_min)

        rgb = rgb / 255.0
        max_room_x = np.max(xyz[:, 0])
        max_room_y = np.max(xyz[:, 1])
        max_room_z = np.max(xyz[:, 2])

        print(f"name:{dataset}\nxyz:{xyz.shape}\nintensity:{intensity.shape}\nrgb:{rgb.shape}")

        offsets = [('zero', 0.0), ('half', args.block_size / 2)]
        for offset_name, offset in offsets:
            idx_h5 = 0
            idx = 0

            print(f'{datetime.now()}-Computing block id of {xyzirgb.shape[0]} points...')
            xyz_min = np.amin(xyz, axis=0, keepdims=True) - offset
            xyz_max = np.amax(xyz, axis=0, keepdims=True)
            block_size = (args.block_size, args.block_size, 2 * (xyz_max[0, -1] - xyz_min[0, -1]))
            # Note: Don't split over z axis.
            xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int32)

            print(f'{datetime.now()}-Collecting points belong to each block...')
            blocks, point_block_indices, block_point_counts = np.unique(xyz_blocks, return_inverse=True,
                                                                        return_counts=True, axis=0)
            block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
            print(f'{datetime.now()}- is split into {blocks.shape[0]} blocks.')

            block_to_block_idx_map = dict()
            for block_idx in range(blocks.shape[0]):
                block = (blocks[block_idx][0], blocks[block_idx][1])
                block_to_block_idx_map[(block[0], block[1])] = block_idx

            # merge small blocks into one of their big neighbors
            block_point_count_threshold = max_num_points / 10
            nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
            block_merge_count = 0
            for block_idx in range(blocks.shape[0]):
                if block_point_counts[block_idx] >= block_point_count_threshold:
                    continue

                block = (blocks[block_idx][0], blocks[block_idx][1])
                for x, y in nbr_block_offsets:
                    nbr_block = (block[0] + x, block[1] + y)
                    if nbr_block not in block_to_block_idx_map:
                        continue

                    nbr_block_idx = block_to_block_idx_map[nbr_block]
                    if block_point_counts[nbr_block_idx] < block_point_count_threshold:
                        continue

                    block_point_indices[nbr_block_idx] = np.concatenate(
                        [block_point_indices[nbr_block_idx], block_point_indices[block_idx]], axis=-1)
                    block_point_indices[block_idx] = np.array([], dtype=np.int32)
                    block_merge_count = block_merge_count + 1
                    break
            print(f'{datetime.now()}-{block_merge_count} of {blocks.shape[0]} blocks are merged.')

            idx_last_non_empty_block = 0
            for block_idx in reversed(range(blocks.shape[0])):
                if block_point_indices[block_idx].shape[0] != 0:
                    idx_last_non_empty_block = block_idx
                    break

            # uniformly sample each block
            for block_idx in range(idx_last_non_empty_block + 1):
                point_indices = block_point_indices[block_idx]
                if point_indices.shape[0] == 0:
                    continue
                block_points = xyz[point_indices]
                block_min = np.amin(block_points, axis=0, keepdims=True)
                xyz_grids = np.floor((block_points - block_min) / args.grid_size).astype(np.int32)
                grids, point_grid_indices, grid_point_counts = np.unique(xyz_grids, return_inverse=True,
                                                                        return_counts=True, axis=0)
                grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))
                grid_point_count_avg = int(np.average(grid_point_counts))
                point_indices_repeated = []
                for grid_idx in range(grids.shape[0]):
                    point_indices_in_block = grid_point_indices[grid_idx]
                    repeat_num = math.ceil(grid_point_count_avg / point_indices_in_block.shape[0])
                    if repeat_num > 1:
                        point_indices_in_block = np.repeat(point_indices_in_block, repeat_num)
                        np.random.shuffle(point_indices_in_block)
                        point_indices_in_block = point_indices_in_block[:grid_point_count_avg]
                    point_indices_repeated.extend(list(point_indices[point_indices_in_block]))
                block_point_indices[block_idx] = np.array(point_indices_repeated)
                block_point_counts[block_idx] = len(point_indices_repeated)

            for block_idx in range(idx_last_non_empty_block + 1):
                point_indices = block_point_indices[block_idx]
                if point_indices.shape[0] == 0:
                    continue

                block_point_num = point_indices.shape[0]
                block_split_num = int(math.ceil(block_point_num * 1.0 / max_num_points))
                point_num_avg = int(math.ceil(block_point_num * 1.0 / block_split_num))
                point_nums = [point_num_avg] * block_split_num
                point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
                starts = [0] + list(np.cumsum(point_nums))

                # Modified following convensions of PointNet.
                np.random.shuffle(point_indices)
                block_points = xyz[point_indices]
                block_intensity = intensity[point_indices]
                block_rgb = rgb[point_indices]
                block_labels = labels[point_indices]
                x, y, z = np.split(block_points, (1, 2), axis=-1)
                norm_x = x / max_room_x
                norm_y = y / max_room_y
                norm_z = z / max_room_z

                minx = np.min(x)
                miny = np.min(y)
                x = x - (minx + args.block_size / 2)
                y = y - (miny + args.block_size / 2)

                block_xyzirgb = np.concatenate([x, y, z, block_intensity, block_rgb, norm_x, norm_y, norm_z], axis=-1)
                for block_split_idx in range(block_split_num):
                    #print(block_split_num, point_nums)
                    start = starts[block_split_idx]
                    point_num = point_nums[block_split_idx]

                    end = start + point_num
                    #print(start, point_num, block_xyzrgb[start:end, :].shape)
                    idx_in_batch = idx % batch_size
                    data[idx_in_batch, 0:point_num, ...] = block_xyzirgb[start:end, :]
                    data_num[idx_in_batch] = point_num
                    label[idx_in_batch] = dataset_idx   # won't be used...
                    label_seg[idx_in_batch, 0:point_num] = block_labels[start:end]
                    indices_split_to_full[idx_in_batch, 0:point_num] = point_indices[start:end]

                    if ((idx + 1) % batch_size == 0) or \
                            (block_idx == idx_last_non_empty_block and block_split_idx == block_split_num - 1):
                        item_num = idx_in_batch + 1
                        filename_h5 = os.path.join(default_h5dir, f'{offset_name}_{idx_h5:d}.h5')
                        print(f'{datetime.now()}-Saving {filename_h5}...')

                        file = h5py.File(filename_h5, 'w')
                        file.create_dataset('data', data=data[0:item_num, ...])
                        file.create_dataset('data_num', data=data_num[0:item_num, ...])
                        file.create_dataset('label', data=label[0:item_num, ...])
                        file.create_dataset('label_seg', data=label_seg[0:item_num, ...])
                        file.create_dataset('indices_split_to_full', data=indices_split_to_full[0:item_num, ...])
                        file.close()
                    idx_h5 = idx_h5 + 1
                idx = idx + 1
        open(dataset_marker, 'w').close()
    print(f'{datetime.now()}-Done.')
if __name__ == '__main__':
    main()