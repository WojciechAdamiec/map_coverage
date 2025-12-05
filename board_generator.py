import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]       # bottom-right (largest sum)
    rect[1] = pts[np.argmin(diff)]    # top-right (smallest diff)
    rect[3] = pts[np.argmax(diff)]    # bottom-left (largest diff)

    return rect


def prepare_board(source_image_path: str, output_image_path: str):
    image_bgr = cv2.imread(source_image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image at {source_image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    clicked_points = []
    use_full_image = {"flag": False}

    def on_click(event):
        if event.inaxes != ax_img:
            return
        if use_full_image["flag"]:
            return

        if event.xdata is None or event.ydata is None:
            return

        if len(clicked_points) < 4:
            clicked_points.append((event.xdata, event.ydata))

            ax_img.plot(event.xdata, event.ydata, "ro")
            ax_img.text(
                event.xdata + 5,
                event.ydata + 5,
                str(len(clicked_points)),
                color="red",
                fontsize=12,
            )
            fig.canvas.draw_idle()

            if len(clicked_points) == 4:
                plt.close(fig)

    def on_skip(event):
        use_full_image["flag"] = True
        clicked_points.clear()
        plt.close(fig)

    fig, ax_img = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)  # space for button at bottom

    ax_img.imshow(image_rgb)
    ax_img.set_title(
        "Click 4 corners of the table in order around it\n"
        "(e.g. top-left → top-right → bottom-right → bottom-left)\n"
        "OR click 'Skip (use whole image)'"
    )
    ax_img.axis("on")

    ax_button = plt.axes([0.35, 0.05, 0.3, 0.08])
    button = Button(ax_button, "Skip (use whole image)")

    fig.canvas.mpl_connect("button_press_event", on_click)
    button.on_clicked(on_skip)
    plt.show()

    if use_full_image["flag"]:
        pts = np.array(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ],
            dtype="float32",
        )
    else:
        if len(clicked_points) != 4:
            raise RuntimeError(
                f"Expected 4 points or skip; got {len(clicked_points)}. Try again."
            )
        pts = np.array(clicked_points, dtype="float32")

    src_rect = order_points(pts)
    (tl, tr, br, bl) = src_rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    square_size = int(max(maxWidth, maxHeight))

    dst_rect = np.array(
        [
            [0, 0],
            [square_size - 1, 0],
            [square_size - 1, square_size - 1],
            [0, square_size - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped_bgr = cv2.warpPerspective(image_bgr, M, (square_size, square_size))

    cv2.imwrite(output_image_path, warped_bgr)
