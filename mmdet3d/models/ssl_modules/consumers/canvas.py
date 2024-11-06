import numpy as np
import cv2
import matplotlib
import os
class Canvas(object):
    def __init__(self, 
                 canvas_shape=(1000, 1000),
                 canvas_x_range=(-50, 50),
                 canvas_y_range=(-50, 50),
                 canvas_bg_color=(0, 0, 0),
                 box_line_thickness=2,
                 box_text_size=0.5):
        """
        Canvas coordinates are imshow coordinates. So, "X" is vertical - neg to
            pos is top to down. "Y" is horizontal - neg to pos is left to right
        """
        
        self.canvas_shape = canvas_shape
        self.canvas_x_range = canvas_x_range
        self.canvas_y_range = canvas_y_range
        self.canvas_bg_color = canvas_bg_color
        self.box_line_thickness = box_line_thickness
        self.box_text_size = box_text_size
        
        self.canvas = np.zeros((*self.canvas_shape, 3), dtype=np.uint8)
        self.canvas[..., :] = canvas_bg_color
        self.draw_origin_arrows()
    
    def get_canvas(self):
        return self.canvas

    def clear_canvas(self):
        self.canvas = np.zeros((*self.canvas_shape, 3), dtype=np.uint8)
        self.canvas[..., :] = canvas_bg_color
        self.draw_origin_arrows()
        
    def get_canvas_coords(self, xy):
        """
        Args:
            xy: ndarray (N, 2) of coordinates
        
        Returns:
            canvas_xy: ndarray (N, 2) of xy scaled into canvas coordinates
            valid_mask: ndarray (N, ) boolean mask indicating which of 
                canvas_xy fits into canvas.
                Invalid locations of canvas_xy are clipped into range.
        """
        xy = np.copy(xy) # prevent in-place modifications

        x = xy[:, 0]
        y = xy[:, 1]

        # Get valid mask
        valid_mask = ((x > self.canvas_x_range[0]) & 
                      (x < self.canvas_x_range[1]) &
                      (y > self.canvas_y_range[0]) & 
                      (y < self.canvas_y_range[1]))
        
        # Rescale points
        x = ((x - self.canvas_x_range[0]) / 
             (self.canvas_x_range[1] - self.canvas_x_range[0]))
        x = x * self.canvas_shape[0]
        x = np.clip(np.around(x), 0, 
                    self.canvas_shape[0] - 1).astype(np.int32)
                    
        y = ((y - self.canvas_y_range[0]) / 
             (self.canvas_y_range[1] - self.canvas_y_range[0]))
        y = y * self.canvas_shape[1]
        y = np.clip(np.around(y), 0, 
                    self.canvas_shape[1] - 1).astype(np.int32)
        
        # Return
        canvas_xy = np.stack([x, y], axis=1)

        return canvas_xy, valid_mask

    def draw_origin_arrows(self):
        """
        Draws a red line in positive x dir, green line in positive y dir from
        true center.
        """
        center = self.get_canvas_coords(np.zeros((1, 2)))[0][0]
        x_len = int(self.canvas_shape[0] / 50)
        y_len = int(self.canvas_shape[1] / 50)

        # Careful - cv2's coordinates is horizontal x vertical, which is
        # transpose compared to regular numpy indexing

        cv2_center = center[::-1].astype(np.int32).tolist()

        # Positive X dir
        self.canvas = cv2.arrowedLine(self.canvas,
                                      tuple(cv2_center),
                                      (cv2_center[0], cv2_center[1] + x_len),
                                      (255, 0, 0),
                                      2)

        # Positive Y dir
        self.canvas = cv2.arrowedLine(self.canvas,
                                      tuple(cv2_center),
                                      (cv2_center[0] + y_len, cv2_center[1]),
                                      (0, 0, 255),
                                      2)

    def draw_lines(self, x=[], y=[], color=(128, 0, 0)):
        """
        It's often useful to have indicator lines denoting x and y distances.
        These are in true coordinates (not canvas)
        """
        # Get canvas equivalents of x and y
        canvas_x = np.zeros((len(x), 2))
        canvas_x[:, 0] = x
        canvas_x, _ = self.get_canvas_coords(canvas_x)
        canvas_x = canvas_x[:, 0]

        canvas_y = np.zeros((len(y), 2))
        canvas_y[:, 1] = y
        canvas_y, _ = self.get_canvas_coords(canvas_y)
        canvas_y = canvas_y[:, 1]

        # Draw lines
        for i, cx in enumerate(canvas_x):
            self.canvas = cv2.line(self.canvas,
                                   (0, int(cx)),
                                   (self.canvas_shape[1] - 1, int(cx)),
                                   color,
                                   2)
            self.canvas = cv2.putText(self.canvas,
                                      str(x[i]),
                                      (0, int(cx)),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      color,
                                      2)

        for i, cy in enumerate(canvas_y):
            self.canvas = cv2.line(self.canvas,
                                   (int(cy), 0),
                                   (int(cy), self.canvas_shape[0] - 1),
                                   color,
                                   2)
            self.canvas = cv2.putText(self.canvas,
                                      str(y[i]),
                                      (int(cy), 10),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      color,
                                      2)
                                      

    def draw_canvas_points(self, 
                           canvas_xy,
                           cmap=None):
        """
        Draws canvas_xy onto self.canvas.
        Args:
            canvas_xy: ndarray (N, 2) of valid canvas coordinates
            cmap: Optionally, a matplotlib cmap can be passed in to color 
                points. Normalized distance from center of canvas will be used 
                as input. Default: None, just white points for all.
                "Spectral" is decent for black background.
        """

        if cmap is None:
            cmap = (lambda d: np.full((*d.shape, 4), fill_value=1))
        else:
            cmap = matplotlib.cm.get_cmap(cmap)

        # Get normalized distances from canvas center
        canvas_center = np.array(self.canvas_shape) / 2
        distances = np.sqrt(((canvas_xy - canvas_center) ** 2).sum(axis=1))
        norm_distances = distances / np.sqrt((canvas_center ** 2).sum())
        
        # Get cmap colors - note that cmap returns (*input_shape, 4), with
        # colors scaled 0 ~ 1
        colors = (cmap(norm_distances)[:, :3] * 255).astype(np.uint8)

        # Place in canvas
        self.canvas[canvas_xy[:, 0], canvas_xy[:, 1]] = colors

    def draw_boxes(self,
                   boxes,
                   colors=(255, 255, 255),
                   texts=None):
        """
        Draws a set of boxes onto the canvas.
        Args:
            boxes: ndarray. Can either be of shape:
                (N, 7): Then, assumes (x, y, z, x_size, y_size, z_size, yaw)
                (N, 5): Then, assumes (x, y, x_size, y_size, yaw)
                Everything is in the same coordinate system as points above 
                (not canvas coordinates)
            colors: Can be either:
                A 3-tuple denoting color. Defaults to white. RGB.
                ndarray of shape (N, 3), of type uint8 representing color
                    of each box.
            texts: list of strings of length N to write next to boxes.
        """
        boxes = np.copy(boxes) # prevent in-place modifications
        assert len(boxes.shape) == 2
        
        if boxes.shape[-1] == 7:
            boxes = boxes[:, [0, 1, 3, 4, 6]]
        else:
            assert boxes.shape[-1] == 5
        
        ## Get the BEV four corners
        # Get BEV 4 corners in box canonical coordinates
        bev_corners = np.array([[
            [0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, -0.5],
            [0.5, -0.5]
        ]]) * boxes[:, None, [2, 3]] # N x 4 x 2

        # Get rotation matrix from yaw
        rot_sin = np.sin(boxes[:, -1])
        rot_cos = np.cos(boxes[:, -1])
        rot_matrix = np.stack([
            [rot_cos, -rot_sin],
            [rot_sin, rot_cos]
        ]) # 2 x 2 x N

        # Rotate BEV 4 corners. Result: N x 4 x 2
        bev_corners = np.einsum('aij,jka->aik', bev_corners, rot_matrix)

        # Translate BEV 4 corners
        bev_corners = bev_corners + boxes[:, None, [0, 1]]

        ## Transform BEV 4 corners to canvas coords
        bev_corners_canvas, valid_mask = \
            self.get_canvas_coords(bev_corners.reshape(-1, 2))
        bev_corners_canvas = bev_corners_canvas.reshape(*bev_corners.shape)
        valid_mask = valid_mask.reshape(*bev_corners.shape[:-1])

        # All four corners need to be within canvas to draw it
        valid_mask = valid_mask.all(axis=1)
        bev_corners_canvas = bev_corners_canvas[valid_mask]
        texts = np.array(texts)[valid_mask]

        ## Draw onto canvas
        # Setup colors
        if isinstance(colors, tuple):
            assert len(colors) == 3
            colors = [colors for _ in range(len(bev_corners_canvas))]
        else:
            assert isinstance(colors, np.ndarray) and colors.dtype == np.uint8
            colors = list(map(tuple, colors.tolist()))

        # Draw the outer boundaries
        idx_draw_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, curr_box_corners in enumerate(bev_corners_canvas):
            curr_box_corners = curr_box_corners.astype(np.int32)
            for start, end in idx_draw_pairs:
                self.canvas = cv2.line(self.canvas,
                                       tuple(curr_box_corners[start][::-1]\
                                        .tolist()),
                                       tuple(curr_box_corners[end][::-1]\
                                        .tolist()),
                                       color=colors[i],
                                       thickness=self.box_line_thickness)
            if texts is not None:
                self.canvas = cv2.putText(self.canvas,
                                          str(texts[i]),
                                          tuple(curr_box_corners[0][::-1]\
                                            .tolist()),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          self.box_text_size,
                                          colors[i],
                                          thickness=self.box_line_thickness)

            # Draw Center Line
            box_center = curr_box_corners.mean(axis=0).astype(np.int32)
            head_center = ((curr_box_corners[0] + curr_box_corners[1]) / 2).astype(np.int32)
            # tail_center = ((curr_box_corners[2] + curr_box_corners[3]) / 2).astype(np.int32)
            self.canvas = cv2.line(self.canvas,
                                    tuple(head_center[::-1].tolist()),
                                    tuple(box_center[::-1].tolist()),
                                    color=colors[i],
                                    thickness=self.box_line_thickness)


    def draw_3d_box(self,image, box_params,lidar2img,sample_idxs):
        def get_3d_box_corners(box):
            x, y, z, length ,width,height, yaw = box
            corners = np.array([
                [-length / 2, -width / 2, 0],
                [length / 2, -width / 2, 0],
                [length / 2, width / 2, 0],
                [-length / 2, width / 2, 0],
                [-length / 2, -width / 2, -height],
                [length / 2, -width / 2, -height],
                [length / 2, width / 2, -height],
                [-length / 2, width / 2, -height]
            ])

            # 旋转
            rot_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            corners = corners @ rot_matrix.T
            corners += np.array([x, y, z+1.5])
            return corners

        for box in box_params:
            corners = get_3d_box_corners(box)

            # 将 3D 点投影到 2D 图像上
            projected_corners = []
            for corner in corners:
                proj_point = lidar2img @ np.append(corner, 1)  # 齐次坐标
                proj_point /= proj_point[2]  # 齐次坐标归一化
                projected_corners.append(proj_point[:2])

            projected_corners = np.array(projected_corners).astype(np.int32)

            # 绘制边界
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 上面四个角
                (4, 5), (5, 6), (6, 7), (7, 4),  # 下面四个角
                (0, 4), (1, 5), (2, 6), (3, 7)  # 上下连接
            ]

            for start, end in edges:
                cv2.line(image, tuple(projected_corners[start]), tuple(projected_corners[end]), (0, 255, 0), thickness=2)
        sample_idx=sample_idxs
        output_path = "/home/lab/SS/environment/vis_img_pvrcnn/"
        save_path = os.path.join(output_path, f"sample_{sample_idx}.png")
        if cv2.imwrite(save_path, image):
            print("Image saved successfully.")
