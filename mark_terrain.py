import cv2
import numpy as np


WINDOW_NAME = "Mark terrain - LMB: add point | N: new piece | Q: finish"
# ------------------------------

def mark_terrain(source_image_path, output_image_path) -> float:
    global display_img, current_polygon, polygons, mask
    img = cv2.imread(source_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {source_image_path}")

    # global state for the mouse callback
    display_img = img.copy()
    polygons = []          # list of list-of-points
    current_polygon = []   # points of polygon being drawn
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # will hold all terrain areas


    def redraw():
        """Redraw overlay of all polygons + current polygon."""
        global display_img
        display_img = img.copy()

        # draw existing polygons
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(display_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.fillPoly(display_img, [pts], color=(0, 0, 255))  # filled (semi-hidden by alpha later)

        # draw current polygon
        if len(current_polygon) > 0:
            pts = np.array(current_polygon, dtype=np.int32)
            cv2.polylines(display_img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
            for p in current_polygon:
                cv2.circle(display_img, p, 3, (0, 255, 0), -1)


    def mouse_callback(event, x, y, flags, param):
        """Handle left-click to add points."""
        global current_polygon, display_img

        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
            redraw()


    def finalize_current_polygon():
        """Add current_polygon to polygons & draw it into the mask."""
        global current_polygon, polygons, mask

        if len(current_polygon) >= 3:
            polygons.append(current_polygon.copy())
            pts = np.array(current_polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        current_polygon = []
        redraw()


    # ----- main interaction loop -----
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 1000)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    redraw()

    while True:
        # show image with 50% alpha overlay of mask for feedback
        overlay = display_img.copy()
        terrain_color = np.zeros_like(img)
        terrain_color[:, :] = (0, 0, 255)  # red overlay for terrain
        terrain_alpha = 0.4

        terrain_mask_bgr = cv2.merge([mask, mask, mask])
        # show image with 50% alpha overlay of mask for feedback
        overlay = display_img.copy()

        terrain_color = np.zeros_like(img)
        terrain_color[:, :] = (0, 0, 255)  # red overlay
        terrain_alpha = 0.4

        # mask as single-channel, expand to 3 channels
        mask3 = cv2.merge([mask, mask, mask])  # (H,W,3)

        # boolean mask where mask > 0
        mask_bool = mask3 > 0

        # apply alpha blending only where terrain exists
        overlay[mask_bool] = (
            terrain_alpha * terrain_color[mask_bool] +
            (1 - terrain_alpha) * overlay[mask_bool]
        ).astype(np.uint8)

        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n'):      # finish current piece, start a new one
            finalize_current_polygon()
        elif key == ord('u'):    # undo last polygon
            if polygons:
                polygons.pop()
                mask[:] = 0
                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                redraw()
        elif key == 27 or key == ord('q'):   # ESC or 'q' to quit and compute
            finalize_current_polygon()
            break

    cv2.destroyAllWindows()

    # ----- compute coverage -----
    terrain_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    coverage = 100.0 * terrain_pixels / total_pixels

    debug_img = img.copy()

    # fill polygons in red (you can change color)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(debug_img, [pts], (0, 0, 255))  # red fill

    # also draw outlines if you want
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(debug_img, [pts], True, (255, 255, 255), 2)  # white border

    # save debug image
    cv2.imwrite(output_image_path, debug_img)

    return coverage
