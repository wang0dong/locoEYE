#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import loguru
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import datetime
from sklearn.cluster import KMeans

LOG = loguru.logger


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        if embedding_image_feats.shape[0] == 0:
            LOG.error("No features to cluster - empty embedding_image_feats")
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret

        db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        # print(f"eps:{self._cfg.POSTPROCESS.DBSCAN_EPS}, minimal samples:{self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES}")

        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            LOG.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            print("[WARN] DBSCAN returned no clusters.")
            return np.zeros_like(mask), [], np.array([])

        lane_coords = []
        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            # mask[pix_coord_idx] = self._color_map[index]
            mask[pix_coord_idx] = self._color_map[index % len(self._color_map)]
            lane_coords.append(coord[idx])

        return mask, lane_coords, db_labels


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, cfg, ipm_remap_file_path=None):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        # assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)
        self._ipm_remap_file_path = ipm_remap_file_path

        if ipm_remap_file_path is not None:
            assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)
            remap_file_load_ret = self._load_remap_matrix()
            self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
            self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']
        else:
            self._remap_to_ipm_x = None
            self._remap_to_ipm_y = None

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def is_valid_segment(self, coords, instance_seg_result, min_points=30, min_vertical_span=0.05, min_slope_threshold=0.01):
        if coords is None or len(coords) < min_points:
            return False
        # Get image dimensions
        h, w = instance_seg_result.shape[:2]  # supports both 2D and 3D arrays

        try:
            norm_x = coords[:, 0] * w
            norm_y = coords[:, 1] * h

            vertical_span = norm_y.max() - norm_y.min()
            if vertical_span < h * min_vertical_span:
                return False

            fit = np.polyfit(norm_y, norm_x, deg=1)
            slope = fit[0]
            if abs(slope) < min_slope_threshold:
                return False
        except:
            return False

        return True

    def merge_similar_lines(self, lane_coords, slope_tol=0.1, intercept_tol=20):
        """
        Merge segments that have similar linear fits (i.e., same rail but different portions).

        Args:
            lane_coords (list[np.ndarray]): List of (N, 2) coords for each segment.
            slope_tol (float): Allowed slope difference.
            intercept_tol (float): Allowed intercept difference.

        Returns:
            merged_coords: list[np.ndarray] with merged segments.
        """
        fit_params = []
        for coords in lane_coords:
            if coords.shape[0] < 2:
                fit_params.append(None)
                continue
            y = coords[:, 1]
            x = coords[:, 0]
            try:
                slope, intercept = np.polyfit(y, x, 1)
                fit_params.append((slope, intercept))
            except:
                fit_params.append(None)

        merged = []
        used = set()

        for i, (c1, p1) in enumerate(zip(lane_coords, fit_params)):
            if i in used or p1 is None:
                continue
            group = [c1]
            used.add(i)
            for j, (c2, p2) in enumerate(zip(lane_coords, fit_params)):
                if j == i or j in used or p2 is None:
                    continue
                if abs(p1[0] - p2[0]) < slope_tol and abs(p1[1] - p2[1]) < intercept_tol:
                    group.append(c2)
                    used.add(j)
            merged.append(np.vstack(group))

        return merged

    def split_multi_track_segment(self, coords, width_thresh=30, min_points=10, y_spread_thresh=20):
        """
        Split a wide/multi-rail segment using 2D KMeans clustering.

        Args:
            coords (np.ndarray): (N, 2) array of (x, y) coordinates.
            width_thresh (float): Minimum x-width to consider splitting.
            min_points (int): Minimum number of points for valid split.
            y_spread_thresh (float): Minimum y-range to avoid vertical clusters.

        Returns:
            List[np.ndarray]: List of 1 or 2 clusters.
        """
        if coords.shape[0] < min_points:
            return [coords]

        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()

        if x_range < width_thresh and y_range < y_spread_thresh:
            return [coords]

        try:
            km = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(coords)
            labels = km.labels_
            cluster1 = coords[labels == 0]
            cluster2 = coords[labels == 1]

            if len(cluster1) < min_points or len(cluster2) < min_points:
                return [coords]

            # Optional: check if line fits are distinct enough
            def line_params(c):
                y, x = c[:, 1], c[:, 0]
                return np.polyfit(y, x, 1)

            slope1, _ = line_params(cluster1)
            slope2, _ = line_params(cluster2)

            if abs(slope1 - slope2) < 0.05:  # they’re both straight and nearly parallel
                return [cluster1, cluster2]

            return [cluster1, cluster2]
        except Exception as e:
            print(f"[WARN] 2D KMeans split failed: {e}")
            return [coords]
        
    def remove_linear_and_duplicates(self, instance_mask, lane_coords, db_labels, slope_tol=0.1, intercept_tol=20):
        """
        Remove one linear rail segment and its duplicates (based on similar line fit).
        
        Args:
            instance_mask (np.ndarray): RGB mask image where each instance has a unique color.
            lane_coords (list[np.ndarray]): List of (N, 2) arrays of coordinates for each segment.
            db_labels (np.ndarray): Cluster labels from DBSCAN.
            slope_tol (float): Slope similarity threshold.
            intercept_tol (float): Intercept similarity threshold.

        Returns:
            filtered_instance_mask, filtered_lane_coords, filtered_db_labels
        """
        h, w, _ = instance_mask.shape
        fit_params = []

        # Step 1: Fit lines to all segments
        for coords in lane_coords:
            if coords.shape[0] < 2:
                fit_params.append(None)
                continue
            y = coords[:, 1]
            x = coords[:, 0]
            try:
                slope, intercept = np.polyfit(y, x, 1)
                fit_params.append((slope, intercept))
            except Exception as e:
                print(f"[WARN] Polyfit failed: {e}")
                fit_params.append(None)

        # Step 2: Find the first reasonably linear segment
        base_idx = None
        for idx, params in enumerate(fit_params):
            if params is not None:
                base_idx = idx
                break

        if base_idx is None:
            return instance_mask, lane_coords, db_labels  # no valid lines to process

        base_slope, base_intercept = fit_params[base_idx]
        print(f"[DEBUG] base segment idx: {base_idx}, slope: {base_slope:.4f}, intercept: {base_intercept:.2f}")


        # Step 3: Identify duplicates (same slope and intercept within tolerance)
        duplicate_indices = []
        print(f"[DEBUG] Checking duplicates with slope_tol={slope_tol}, intercept_tol={intercept_tol}")

        for idx, params in enumerate(fit_params):
            if idx == base_idx or params is None:
                continue
            slope, intercept = params
            print(f"[DEBUG] Segment {idx}: slope={slope:.4f}, intercept={intercept:.2f}")
            if abs(slope - base_slope) < slope_tol and abs(intercept - base_intercept) < intercept_tol:
                duplicate_indices.append(idx)

        remove_indices = [base_idx] + duplicate_indices
        print(f"[DEBUG] Segments to remove: {remove_indices}")

        # Step 4: Remove from mask
        mask_removed = instance_mask.copy()
        for idx in enumerate(lane_coords):
            if idx in remove_indices:
                coords = lane_coords[idx]
                # Remove this segment from mask
                x = np.clip(coords[:, 0].astype(np.int32), 0, w - 1)
                y = np.clip(coords[:, 1].astype(np.int32), 0, h - 1)
                mask_removed[y, x] = [0, 0, 0]  # black out

        # Step 5: Filter out removed segments from coords and labels
        filtered_coords = []
        filtered_labels = []
        for idx, coords in enumerate(lane_coords):
            if idx not in remove_indices:
                filtered_coords.append(coords)
                filtered_labels.append(db_labels[idx])

        filtered_labels = np.array(filtered_labels)
        print(f"[DEBUG] Segments before filtering: {len(lane_coords)}")
        print(f"[DEBUG] Segments after filtering: {len(filtered_coords)}")

        return mask_removed, filtered_coords, filtered_labels

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=10, source_image=None,
                    with_lane_fit=True, data_source='rail',
                    enable_post_processing=True, debug_print=True):
        """
        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param with_lane_fit:
        :param data_source:
        :return:
        """

        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # === PREPROCESS to split merged rails before morphology ===
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        # Ensure binary_seg_result is uint8 binary (0 or 255)
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)
        _, binary_mask = cv2.threshold(binary_seg_result, 127, 255, cv2.THRESH_BINARY)

        # Compute distance transform
        distance = ndi.distance_transform_edt(binary_mask)
        # Get coordinates of local maxima
        coordinates = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary_mask)        
        # Create boolean mask from coordinates
        local_maxi = np.zeros_like(distance, dtype=bool)
        local_maxi[tuple(coordinates.T)] = True
        markers = ndi.label(local_maxi)[0]

        # Step 3: Watershed to split merged blobs
        labels_ws = watershed(-distance, markers, mask=binary_mask)

        # Optional: convert watershed output back to binary for morphology
        watershed_mask = (labels_ws > 0).astype(np.uint8) * 255

        # === CONTINUE YOUR ORIGINAL PIPELINE ===


        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        print("[DEBUG] Morphological mask non-zero count:", np.count_nonzero(morphological_ret))
        print("[DEBUG] Instance embedding shape:", instance_seg_result.shape if instance_seg_result is not None else "None")

        # apply embedding features cluster
        cluster_result = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        # Gracefully handle return if fewer than 3 items
        if cluster_result is None or len(cluster_result) != 3:
            print("[WARN] Clustering failed or skipped - skipping frame.")
            return {
                'mask_image': None,
                'lane_coords': [],
                'cluster_ids': []
            }

        mask_image, lane_coords, cluster_ids = cluster_result

        # Split wide/multi-track segments BEFORE clustering
        split_coords = []
        for coords in lane_coords:
            segments = self.split_multi_track_segment(coords)
            split_coords.extend(segments)

        # Replace original coords with split ones
        lane_coords = split_coords

        # Step 3: Merge same-rail segments (optional but helps with oversegmentation)
        lane_coords = self.merge_similar_lines(lane_coords)

        # Remove one linear segment and its duplicates
        if mask_image is not None and lane_coords is not None and cluster_ids is not None:
            mask_image, lane_coords, cluster_ids = self.remove_linear_and_duplicates(
            instance_mask=mask_image,
            lane_coords=lane_coords,
            db_labels=cluster_ids,
            slope_tol=0.1,          # tweak these
            intercept_tol=20        # tweak these
        )
        print(f"[DEBUG] Segments after linear+dup filtering: {len(lane_coords)}")
        for i, coords in enumerate(lane_coords):
            print(f"[DEBUG] Segment {i}: #points = {len(coords)}")

        if not enable_post_processing:
            mask_image = binary_seg_result.copy()
            return {
                'mask_image': mask_image,
                'fit_params': None,
                'source_image': source_image
            }

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }
        if not with_lane_fit:
            tmp_mask = cv2.resize(
                mask_image,
                dsize=(source_image.shape[1], source_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            source_image = cv2.addWeighted(source_image, 0.6, tmp_mask, 0.4, 0.0, dst=source_image)
            return {
                'mask_image': mask_image,
                'fit_params': None,
                'source_image': source_image,
            }

        # lane line fit
        h, w = binary_seg_result.shape
        
        fit_params = []
        src_lane_pts = []  # lane pts every single lane

        for lane_index, coords in enumerate(lane_coords):

            print(f"[DEBUG] Processing lane {lane_index}, #points: {len(coords)}")

            if not self.is_valid_segment(coords, instance_seg_result):
                continue  # Skip mislabelled or invalid segments

            tmp_mask = np.zeros((h, w), dtype=np.uint8)
            tmp_mask[tuple((np.int_(coords[:, 1] * h / instance_seg_result.shape[0]),
                            np.int_(coords[:, 0] * w / instance_seg_result.shape[1])))] = 255

            tmp_ipm_mask = tmp_mask
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

            # Linear fit (degree = 1)
            fit_param = np.polyfit(nonzero_y, nonzero_x, 1) # y = m*x + c
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(nonzero_y.min(), nonzero_y.max(), num=tmp_mask.shape[0])
            # fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2] # quadratic 
            fit_x = fit_param[0] * plot_y + fit_param[1] # linear

            # Scale both points and fit back to original image size
            scale_x = source_image.shape[1] / tmp_mask.shape[1]
            scale_y = source_image.shape[0] / tmp_mask.shape[0]

            # Scale points
            nonzero_x_scaled = nonzero_x * scale_x
            nonzero_y_scaled = nonzero_y * scale_y
            fit_x_scaled = fit_x * scale_x
            plot_y_scaled = plot_y * scale_y

            if debug_print:
                # Plot correctly oriented image with proper scale
                plt.figure(figsize=(10, 6))
                plt.imshow(source_image)
                plt.scatter(nonzero_x_scaled, nonzero_y_scaled, s=2, color='yellow', label='Rail pixels')  # points
                plt.plot(fit_x_scaled, plot_y_scaled, color='red', linewidth=2, label='Fitted line')       # line
                label_x = fit_x_scaled[0]   # x position for label (scaled)
                label_y = plot_y_scaled[0]  # y position for label (scaled)
                plt.text(label_x, label_y, f'Segment {lane_index}', color='cyan', fontsize=12, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=2))
                plt.title("Corrected Fitted Line on Original Image")
                plt.legend()
                plt.axis("off")
                plt.gca().invert_yaxis()  # Match image coordinate system (origin at top-left)

                # Save to file
                OUTPUT_DIR = './Tracks/output/predictions/debug'
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(OUTPUT_DIR, f'{timestamp}_fitted_segment_{lane_index}.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                # x_coords = fit_x
                # y_coords = plot_y
                # for x, y in zip(x_coords, y_coords):
                #     if 0 <= int(x) < source_image.shape[1] and 0 <= int(y) < source_image.shape[0]:
                #         cv2.circle(source_image, (int(x), int(y)), 2, (0, 255, 0), -1)

            lane_pts = []
            if self._remap_to_ipm_x is not None and self._remap_to_ipm_y is not None:
                for i in range(0, plot_y.shape[0], 5):
                    y_idx = int(np.clip(plot_y[i], 0, ipm_image_height - 1))
                    x_idx = int(np.clip(fit_x[i], 0, ipm_image_width - 1))

                    src_x = self._remap_to_ipm_x[y_idx, x_idx]
                    if src_x <= 0:
                        continue
                    src_y = max(self._remap_to_ipm_y[y_idx, x_idx], 0)
                    lane_pts.append([src_x, src_y])
            else:
                for i in range(0, plot_y.shape[0], 5):
                    src_x = fit_x[i] * (source_image.shape[1] / tmp_mask.shape[1])
                    src_y = plot_y[i] * (source_image.shape[0] / tmp_mask.shape[0])
                    lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)

        source_image_width = source_image.shape[1]
        source_image_height = source_image.shape[0]

        # print("source_image:", source_image.shape)
        # print("mask_image:", mask_image.shape)
        # print("tmp_mask used for fitting:", tmp_mask.shape)

        for index, lane_pts in enumerate(src_lane_pts):
            if len(lane_pts) < 2:
                continue  # not enough points to draw a line

            lane_color = self._color_map[index].tolist()

            roi_y_start = source_image_height // 2  # start of lower half
            roi_y_end = source_image_height       # bottom of image
            # Draw connected lines between points for a visible lane
            for i in range(1, len(lane_pts)):
                pt1 = tuple(np.int32(lane_pts[i - 1]))
                pt2 = tuple(np.int32(lane_pts[i]))

                # Only draw if either point is within ROI vertically
                if (roi_y_start <= pt1[1] < roi_y_end or roi_y_start <= pt2[1] < roi_y_end) and \
                all(0 <= c < source_image.shape[1] for c in (pt1[0], pt2[0])) and \
                all(0 <= r < source_image.shape[0] for r in (pt1[1], pt2[1])):
                    cv2.line(source_image, pt1, pt2, lane_color, thickness=2)

            for pt in lane_pts:
                x, y = map(int, pt)
                if roi_y_start <= y < roi_y_end:
                    cv2.circle(source_image, (x, y), 3, lane_color, -1)

        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
            'cluster_ids': cluster_ids,
        }

        return ret
