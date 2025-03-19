import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle
from numpy.polynomial.polynomial import Polynomial
from matplotlib.widgets import Slider, Button
from segment_anything import sam_model_registry, SamPredictor

def rotate_with_slider(image):
    angle = [0]  # mutable so the button can access latest value

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img_display = ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Add dotted midlines for alignment
    h, w = image.shape[:2]
    ax.axhline(h // 2, color='white', linestyle='--', linewidth=0.7)
    ax.axvline(w // 2, color='white', linestyle='--', linewidth=0.7)
    ax.set_title("Adjust rotation. Press 'Done' when satisfied.")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Angle', -180, 180, valinit=0)

    ax_button = plt.axes([0.45, 0.02, 0.1, 0.04])
    button = Button(ax_button, 'Done')

    def update(val):
        deg = slider.val
        angle[0] = deg
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        img_display.set_data(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_button_clicked(event):
        plt.close()

    button.on_clicked(on_button_clicked)
    plt.show()

    final_angle = angle[0]
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
    final_rotated = cv2.warpAffine(image, M, (w, h))
    return final_rotated

# --- Step 1: Load summary and get best image ---
def get_best_image_path(deployment_path):
    deployment_name = os.path.basename(deployment_path)
    summary_csv = [f for f in os.listdir(deployment_path) if f.endswith("_summary.csv")][0]
    summary_df = pd.read_csv(os.path.join(deployment_path, summary_csv))
    
    best_row = summary_df[summary_df["Best"] == "T"].iloc[0]
    photo_id = str(best_row["frames"])
    
    # Construct filename like: UAV21_196e_1_5559_undist.jpg
    image_filename = f"{deployment_name}_1_{photo_id}_undist.jpg"
    image_path = os.path.join(deployment_path, "corrected_frames", image_filename)
    return image_path


def crop_and_rotate_fluke(image_path):
    img = cv2.imread(image_path)

    # Resize for ROI selection
    screen_scale = 1000 / img.shape[1]
    display_img = cv2.resize(img, (0, 0), fx=screen_scale, fy=screen_scale) if screen_scale < 1.0 else img.copy()

    # Select fluke area
    roi_scaled = cv2.selectROI("Select fluke area", display_img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = [int(v / screen_scale) for v in roi_scaled]
    cropped = img[y:y+h, x:x+w]

    # NEW: Use interactive slider to rotate
    rotated = rotate_with_slider(cropped)

    return rotated

# --- Step 3: Manual trace + 11-section processing ---
def interactive_threshold_selector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    mask_display = ax.imshow(np.zeros_like(gray), cmap='gray')
    ax.set_title("Adjust threshold to segment fluke")
    ax.axis("off")

    # Sliders
    ax_block = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_c = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_block = Slider(ax_block, 'Block Size', 3, 99, valinit=11, valstep=2)
    slider_c = Slider(ax_c, 'C (bias)', -20, 20, valinit=2, valstep=1)

    # Done button
    ax_button = plt.axes([0.45, 0.05, 0.1, 0.04])
    button = Button(ax_button, 'Done')
    done = {'flag': False, 'mask': None}

    def update(val):
        block = int(slider_block.val)
        C = slider_c.val

        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, C)

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Draw contour
        contour_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 1)

        mask_display.set_data(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
        fig.canvas.draw_idle()
        done['mask'] = mask

    def on_done(event):
        done['flag'] = True
        plt.close()

    slider_block.on_changed(update)
    slider_c.on_changed(update)
    button.on_clicked(on_done)
    update(None)  # Initial render
    plt.show()

    return done['mask']

# --- Step 4: 11-section processing ---
adjusted_axis_x = None


def get_fluke_mask_with_sam(image_bgr, checkpoint_path="/Users/georgesato/PhD/Chapter1/Fluke_Measurements/Processing code/Python/Bose and Lien 1989/segment-anything/sam_vit_b_01ec64.pth"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Display image and ask user for click
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.set_title("Click a point on the fluke")
    clicked = plt.ginput(1, timeout=0)
    plt.close()

    if not clicked:
        print("❌ No point selected. Aborting SAM prediction.")
        return None

    input_point = np.array([clicked[0]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    # Show mask preview with Continue
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(masks[0], cmap='gray')
    ax.set_title("SAM Mask Preview")
    ax.axis("off")

    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn = Button(ax_button, 'Continue')
    btn.on_clicked(lambda event: plt.close())
    plt.show()

    return masks[0].astype(np.uint8) * 255


def process_fluke_image_kmeans(img, output_csv_path, output_img_path, pixel_to_m=0.005):
    global adjusted_axis_x

    h_img, w_img = img.shape[:2]
    scale_factor = min(1000 / w_img, 1.0)
    display_img = cv2.resize(img, (int(w_img * scale_factor), int(h_img * scale_factor)))
    display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    # --- 1. Get fluke mask using SAM ---
    refined_mask = get_fluke_mask_with_sam(img)
    if refined_mask is None:
        return pd.DataFrame()

    # Resize mask to match display
    refined_mask_display = cv2.resize(refined_mask, (display_rgb.shape[1], display_rgb.shape[0]))
    overlay_mask = np.zeros_like(display_rgb)
    overlay_mask[:, :, 1] = refined_mask_display
    blended = cv2.addWeighted(display_rgb, 0.7, overlay_mask, 0.3, 0)

    import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial
from segment_anything import sam_model_registry, SamPredictor

# --- New: adjustable axis (global var for simplicity) ---
adjusted_axis_x = None


def get_fluke_mask_with_sam(image_bgr, checkpoint_path="/Users/georgesato/PhD/Chapter1/Fluke_Measurements/Processing code/Python/Bose and Lien 1989/segment-anything/sam_vit_b_01ec64.pth"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Display image and ask user for click
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.set_title("Click a point on the fluke")
    clicked = plt.ginput(1, timeout=0)
    plt.close()

    if not clicked:
        print("❌ No point selected. Aborting SAM prediction.")
        return None

    input_point = np.array([clicked[0]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    # Show mask preview with Continue
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(masks[0], cmap='gray')
    ax.set_title("SAM Mask Preview")
    ax.axis("off")

    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn = Button(ax_button, 'Continue')
    btn.on_clicked(lambda event: plt.close())
    plt.show()

    return masks[0].astype(np.uint8) * 255


def process_fluke_image_kmeans(img, output_csv_path, output_img_path, pixel_to_m=0.005):
    global adjusted_axis_x

    h_img, w_img = img.shape[:2]
    scale_factor = min(1000 / w_img, 1.0)
    display_img = cv2.resize(img, (int(w_img * scale_factor), int(h_img * scale_factor)))
    display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    # --- 1. Get fluke mask using SAM ---
    refined_mask = get_fluke_mask_with_sam(img)
    if refined_mask is None:
        return pd.DataFrame()

    # Resize mask to match display
    refined_mask_display = cv2.resize(refined_mask, (display_rgb.shape[1], display_rgb.shape[0]))
    overlay_mask = np.zeros_like(display_rgb)
    overlay_mask[:, :, 1] = refined_mask_display
    blended = cv2.addWeighted(display_rgb, 0.7, overlay_mask, 0.3, 0)

    # --- 2. Ask user to adjust symmetry axis with slider ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(blended)
    default_axis = display_rgb.shape[1] // 2
    axis_line = ax.axvline(default_axis, color='cyan', linestyle='-', linewidth=1.5)
    ax.set_title("Step 1: Adjust symmetry axis (then confirm)")

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Axis X', 0, display_rgb.shape[1], valinit=default_axis)

    def update_axis(val):
        axis_line.set_xdata([val, val])
        fig.canvas.draw_idle()

    slider.on_changed(update_axis)

    confirm_ax = plt.axes([0.4, 0.02, 0.2, 0.05])
    confirmed = {'value': False}

    def confirm_callback(event):
        confirmed['value'] = True
        plt.close()

    confirm_btn = Button(confirm_ax, 'Confirm')
    confirm_btn.on_clicked(confirm_callback)
    plt.show()

    if not confirmed['value']:
        return pd.DataFrame()

    adjusted_axis_x = slider.val
    symmetry_axis_px = int(adjusted_axis_x / scale_factor)

    # --- 3. Ask for base and tip click ---
    while True:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(blended)
        ax.axhline(display_rgb.shape[0] / 2, color='white', linestyle='--', linewidth=0.7)
        ax.axvline(adjusted_axis_x, color='cyan', linestyle='-', linewidth=1.5)
        ax.set_title("Step 2: Click fluke base then tip (then confirm or retry)")

        clicked_pts = plt.ginput(2, timeout=0)

        if len(clicked_pts) < 2:
            print("❌ Not enough points selected. Aborting sectioning.")
            plt.close()
            return pd.DataFrame()

        for (cx, cy) in clicked_pts:
            ax.plot(cx, cy, 'rx')
        fig.canvas.draw()

        # Confirmation and retry buttons
        confirm_ax = plt.axes([0.35, 0.02, 0.15, 0.05])
        retry_ax = plt.axes([0.55, 0.02, 0.15, 0.05])
        clicked_confirmed = {'value': False}

        def confirm_callback(event):
            clicked_confirmed['value'] = True
            plt.close()

        def retry_callback(event):
            clicked_confirmed['value'] = False
            plt.close()

        confirm_btn = Button(confirm_ax, 'Confirm')
        retry_btn = Button(retry_ax, 'Retry')
        confirm_btn.on_clicked(confirm_callback)
        retry_btn.on_clicked(retry_callback)
        plt.show()

        if clicked_confirmed['value']:
            break

    (x1, y1), (x2, y2) = [(x / scale_factor, y / scale_factor) for (x, y) in clicked_pts]

    # --- 3. Generate 11 sections ---
    x_edges = np.linspace(x1, x2, 12)
    chord_lengths = []
    span_positions = []
    leading_edges = []
    trailing_edges = []
    section_centers = []
    pitch_pts = []

    overlay = img.copy()
    for i in range(11):
        x_start, x_end = int(x_edges[i]), int(x_edges[i+1])
        section = refined_mask[:, x_start:x_end]
        y_coords = np.where(section > 0)[0]

        if len(y_coords) > 0:
            y_min = int(np.min(y_coords))
            y_max = int(np.max(y_coords))
            chord = (y_max - y_min) * pixel_to_m
            mid_x = int(0.5 * (x_start + x_end))

            leading_edges.append(y_min * pixel_to_m)
            trailing_edges.append(y_max * pixel_to_m)
            chord_lengths.append(chord)
            section_centers.append(mid_x)
            span_pos = (mid_x - x_edges[0]) * pixel_to_m
            span_positions.append(span_pos)

            cv2.line(overlay, (mid_x, y_min), (mid_x, y_max), (0, 0, 255), 2)
            cv2.putText(overlay, f"{i+1}", (mid_x + 5, y_max),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # --- 4. Draw outline & symmetry ---
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    cv2.line(overlay, (symmetry_axis_px, 0), (symmetry_axis_px, h_img), (255, 255, 0), 2)
 
    # --- 5. Pitching axis estimation ---
    chord_ref = [0.87, 0.80, 0.72, 0.64, 0.57, 0.49, 0.42, 0.35, 0.27, 0.20, 0.00]
    pitch_ref = [0.435, 0.365, 0.300, 0.245, 0.190, 0.130, 0.065, 0.000, -0.080, -0.175, 0.000]

    interp_func = interp1d(chord_ref, pitch_ref, kind='linear', bounds_error=False, fill_value="extrapolate")
    poly_fit = Polynomial.fit(chord_ref, pitch_ref, deg=3).convert()

    chord_arr = np.array(chord_lengths)
    b_linear = interp_func(chord_arr)
    b_poly = poly_fit(chord_arr)

    # Add final station (tip) with zero chord
    if len(span_positions) > 0:
        span_positions.append((x_edges[-1] - x_edges[0]) * pixel_to_m)
        chord_lengths.append(0.0)
        leading_edges.append(np.nan)
        trailing_edges.append(np.nan)
        b_linear = np.append(b_linear, [0.0])
        b_poly = np.append(b_poly, [0.0])
        section_centers.append(int(0.5 * (x_edges[-2] + x_edges[-1])))

    for i in range(len(chord_lengths)):
        if chord_lengths[i] > 0:
            cx = section_centers[i]
            py = int((leading_edges[i] / pixel_to_m) + b_poly[i] * ((trailing_edges[i] - leading_edges[i]) / pixel_to_m))
            pitch_pts.append((cx, py))
            cv2.circle(overlay, (cx, py), 3, (255, 0, 255), -1)

    # --- 6. Save pitching fit plot ---
    x = np.array(chord_ref)
    y = np.array(pitch_ref)
    x_fit = np.linspace(0, 0.9, 200)
    plt.figure()
    plt.plot(x, y, 'ko', label='Reference')
    plt.plot(x_fit, interp_func(x_fit), 'r--', label='Linear interp')
    plt.plot(x_fit, poly_fit(x_fit), 'b-', label='Polynomial fit')
    plt.xlabel("Chord length (m)")
    plt.ylabel("Pitching axis (fraction)")
    plt.legend()
    plt.title("Pitching Axis Position Estimation")
    plt.savefig("/Users/georgesato/PhD/Chapter1/Fluke_Measurements/Processing code/Python/Bose and Lien 1989/pitching_axis_fit.png")
    plt.close()

    # --- 7. Compute area and AR ---
    strip_width = (abs(x2 - x1) / 11) * pixel_to_m
    area = np.sum([c * strip_width for c in chord_lengths])
    semi_span = 11 * strip_width
    AR = (4 * semi_span**2) / (2 * area) if area > 0 else np.nan
    mean_chord = np.mean(chord_lengths[:-1])  # exclude tip

    lengths = list(map(len, [leading_edges, trailing_edges, chord_lengths, b_linear, b_poly]))
    if len(set(lengths)) > 1:
        print("❌ Arrays are mismatched in length:", lengths)
        return pd.DataFrame()

    # --- 8. Save CSV ---
    df = pd.DataFrame({
        "station": list(range(1, 12)),
        "leading_edge": leading_edges[:11],
        "trailing_edge": trailing_edges[:11],
        "chord_length": chord_lengths[:11],
        "b_linear": b_linear[:11],
        "b_poly": b_poly[:11]
    })

    df.to_csv(output_csv_path, index=False)

    # --- 9. Save overlay ---
    cv2.imwrite(output_img_path, overlay)

    # --- Preview and confirm final result ---
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Final section overlay – confirm to save or retry")

    confirm_ax = plt.axes([0.35, 0.02, 0.15, 0.05])
    retry_ax = plt.axes([0.55, 0.02, 0.15, 0.05])
    final_confirmed = {'value': False}

    def confirm_overlay(event):
        final_confirmed['value'] = True
        plt.close()

    def retry_overlay(event):
        final_confirmed['value'] = False
        plt.close()

    btn_confirm = Button(confirm_ax, 'Confirm')
    btn_retry = Button(retry_ax, 'Retry')
    btn_confirm.on_clicked(confirm_overlay)
    btn_retry.on_clicked(retry_overlay)
    plt.show()

    if not final_confirmed['value']:
        return process_fluke_image_kmeans(img, output_csv_path, output_img_path, pixel_to_m)

    print(f"✅ Saved: {output_csv_path}\n🖼️ Overlay saved: {output_img_path}")

    return df


# --- MAIN WRAPPER ---
def run_fluke_extraction_for_uav21_196e(base_dir):
    deployment = "UAV22_201d"
    dep_path = os.path.join(base_dir, deployment)
    best_image_path = get_best_image_path(dep_path)
    print(f"📸 Best image selected: {best_image_path}")

    # Crop + rotate
    cropped_img = crop_and_rotate_fluke(best_image_path)
    output_folder = os.path.join(dep_path, "Bose_Lien_Fluke_Dimensions")
    os.makedirs(output_folder, exist_ok=True)

    # Save cropped image
    cropped_img_path = os.path.join(output_folder, "cropped_fluke.jpg")
    cv2.imwrite(cropped_img_path, cropped_img)
    print(f"✅ Saved cropped fluke to: {cropped_img_path}")

    # Extract measurements
    output_csv = os.path.join(output_folder, "fluke_dimensions.csv")
    output_img = os.path.join(output_folder, "fluke_overlay.jpg")
    df = process_fluke_image_kmeans(cropped_img, output_csv, output_img)
    print("🎉 Done!")


# --- Run it ---
run_fluke_extraction_for_uav21_196e("/Users/georgesato/PhD/Chapter1/Fluke_Measurements")
