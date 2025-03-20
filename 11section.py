import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import Polynomial
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

def select_fluke_tip(img, refined_mask, pitch_axis_y, adjusted_axis_x, root_chord_x, scale_factor, pixel_to_m):
    # --- 4: Click fluke tip ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axhline(pitch_axis_y, color='magenta', linestyle='--')
    ax.axvline(adjusted_axis_x, color='cyan', linestyle='--')
    ax.axvline(root_chord_x * scale_factor, color='orange', linestyle='--')
    ax.set_title("Step 4: Click fluke tip")

    tip_point = plt.ginput(1, timeout=0)
    plt.close()

    if not tip_point:
        print("❌ No tip selected. Aborting.")
        return pd.DataFrame()

    (x_tip, y_tip) = tip_point[0]
    x_tip /= scale_factor
    y_tip /= scale_factor

    overlay = img.copy()

    # --- Ensure root_chord_x is to the left of x_tip ---
    if x_tip < root_chord_x:
        root_chord_x, x_tip = x_tip, root_chord_x  # Swap to maintain left → right direction

    # --- Generate 10 sections from root to tip ---
    x_edges = np.linspace(root_chord_x, x_tip, 11)  # 10 segments = 11 edges
    chord_lengths = []
    span_positions = []
    leading_edges = []
    trailing_edges = []
    section_centers = []

    for i in range(10):
        x_start, x_end = int(x_edges[i]), int(x_edges[i + 1])
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
            span_positions.append((mid_x - x_edges[0]) * pixel_to_m)

    # --- Clean up and finalize chord section data ---
    if len(chord_lengths) < 11:
        span_positions.append((x_edges[-1] - x_edges[0]) * pixel_to_m)
        chord_lengths.append(0.0)
        leading_edges.append(np.nan)
        trailing_edges.append(np.nan)
        section_centers.append(int(0.5 * (x_edges[-2] + x_edges[-1])))

    # --- Check lengths before creating DataFrame ---
    lengths = {
        "chord_lengths": len(chord_lengths),
        "leading_edges": len(leading_edges),
        "trailing_edges": len(trailing_edges),
        "section_centers": len(section_centers),
        "pitch_axis_m": 11
    }
    print("✅ Section data lengths:", lengths)

    if len(set(lengths.values())) > 1:
        print("❌ Mismatched lengths detected. Aborting.")
        return pd.DataFrame()

    df = pd.DataFrame({
        "station": list(range(1, 12)),
        "leading_edge": leading_edges,
        "trailing_edge": trailing_edges,
        "chord_length": chord_lengths,
        "pitch_axis_m": [pitch_axis_y] * 11
    })
    
    return df, overlay

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
    ax.set_title("Step 1: Adjust symmetry axis (then confirm)", fontsize=14, weight='bold')

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

    # --- 2.5: Ask user to adjust root chord position (vertical base line) ---
    fig, ax = plt.subplots()
    fig.suptitle("Step 2: Adjust Root Chord (Vertical Base Line)", fontsize=14, weight='bold')
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(blended)
    default_root_x = display_rgb.shape[1] * 0.25
    root_line = ax.axvline(default_root_x, color='orange', linestyle='--', label='Root Chord')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='cyan', linestyle='--', label='Symmetry Axis'),
        Line2D([0], [0], color='orange', linestyle='--', label='Root Chord')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax_slider_root = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider_root = Slider(ax_slider_root, 'Root Chord X', 0, display_rgb.shape[1], valinit=default_root_x)

    root_confirmed = {'value': False}

    def update_root(val):
        root_line.set_xdata([val, val])
        fig.canvas.draw_idle()

    slider_root.on_changed(update_root)

    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    btn_confirm = Button(ax_button, 'Confirm')

    def confirm_root(event):
        root_confirmed['value'] = True
        plt.close()

    btn_confirm.on_clicked(confirm_root)
    plt.show()

    if not root_confirmed['value']:
        return pd.DataFrame()

    root_chord_x = slider_root.val / scale_factor

    # --- 3: Adjust pitching axis BEFORE tip click ---
    fig, ax = plt.subplots()
    fig.suptitle("Step 3: Adjust Pitching Axis Height (Horizontal)", fontsize=14, weight='bold')
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(blended)
    pitch_line_y = display_rgb.shape[0] // 2
    line = ax.axhline(pitch_line_y, color='magenta', linestyle='--', label='Pitch Axis')
    ax.axvline(adjusted_axis_x, color='cyan', linestyle='--', linewidth=1.2)
    ax.axvline(root_chord_x * scale_factor, color='orange', linestyle='--', linewidth=1.2)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='cyan', linestyle='--', label='Symmetry Axis'),
        Line2D([0], [0], color='orange', linestyle='--', label='Root Chord'),
        Line2D([0], [0], color='magenta', linestyle='--', label='Pitching Axis (Adjustable)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax_slider_pitch = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider_pitch = Slider(ax_slider_pitch, 'Pitch Axis Y', 0, display_rgb.shape[0], valinit=pitch_line_y)

    pitch_confirmed = {'value': False}

    def update_pitch(val):
        line.set_ydata([val, val])
        fig.canvas.draw_idle()

    slider_pitch.on_changed(update_pitch)

    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    btn_pitch_confirm = Button(ax_button, 'Confirm')

    def confirm_pitch(event):
        pitch_confirmed['value'] = True
        plt.close()

    btn_pitch_confirm.on_clicked(confirm_pitch)
    plt.show()

    if not pitch_confirmed['value']:
        return pd.DataFrame()

    pitch_axis_y = slider_pitch.val
    pitch_axis_m = pitch_axis_y * pixel_to_m

    # --- 3.5: Add horizontal cutoff to mask (removing peduncle above line) ---
    fig, ax = plt.subplots()
    fig.suptitle("Step 3.5: Adjust Cutoff Line (Removes Peduncle)", fontsize=14, weight='bold')
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(blended)
    cutoff_line = ax.axhline(display_rgb.shape[0] * 0.2, color='red', linestyle='--')

    ax_slider_cut = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider_cut = Slider(ax_slider_cut, 'Cutoff Y', 0, display_rgb.shape[0], valinit=display_rgb.shape[0] * 0.2)

    cutoff_confirmed = {'value': False}

    def update_cutoff(val):
        cutoff_line.set_ydata([val, val])
        fig.canvas.draw_idle()

    slider_cut.on_changed(update_cutoff)

    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    btn_cut = Button(ax_button, 'Confirm')

    def confirm_cutoff(event):
        cutoff_confirmed['value'] = True
        plt.close()

    btn_cut.on_clicked(confirm_cutoff)
    plt.show()

    if not cutoff_confirmed['value']:
        return pd.DataFrame()

    cutoff_y = int(slider_cut.val / scale_factor)
    refined_mask[:cutoff_y, :] = 0  # Zero out mask above the cutoff
    # Recalculate the blended image to reflect cutoff mask
    refined_mask_display = cv2.resize(refined_mask, (display_rgb.shape[1], display_rgb.shape[0]))
    overlay_mask = np.zeros_like(display_rgb)
    overlay_mask[:, :, 1] = refined_mask_display
    blended = cv2.addWeighted(display_rgb, 0.7, overlay_mask, 0.3, 0)

    # --- 4: Click fluke tip ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(blended)
    ax.axhline(pitch_axis_y, color='magenta', linestyle='--')
    ax.axvline(adjusted_axis_x, color='cyan', linestyle='--')
    ax.axvline(root_chord_x * scale_factor, color='orange', linestyle='--')
    ax.set_title("Step 4: Click fluke tip")

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='cyan', linestyle='--', label='Symmetry Axis'),
        Line2D([0], [0], color='orange', linestyle='--', label='Root Chord'),
        Line2D([0], [0], color='magenta', linestyle='--', label='Pitching Axis (Adjustable)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    tip_point = plt.ginput(1, timeout=0)
    plt.close()

    if not tip_point:
        print("❌ No tip selected. Aborting.")
        return pd.DataFrame()

    (x_tip, y_tip) = tip_point[0]
    x_tip /= scale_factor
    y_tip /= scale_factor

    overlay = img.copy()

    # --- Ensure root_chord_x is to the left of x_tip ---
    if x_tip < root_chord_x:
        root_chord_x, x_tip = x_tip, root_chord_x  # Swap to maintain left → right direction

    # --- Generate 10 sections from root to tip ---
    x_edges = np.linspace(root_chord_x, x_tip, 11)  # 10 segments = 11 edges
    chord_lengths = []
    span_positions = []
    leading_edges = []
    trailing_edges = []
    section_centers = []

    for i in range(10):
        x_start, x_end = int(x_edges[i]), int(x_edges[i + 1])
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
            span_positions.append((mid_x - x_edges[0]) * pixel_to_m)

    # --- Clean up and finalise chord section data ---

    # Ensure we only have 11 stations (10 + tip)
    if len(chord_lengths) < 11:
        # Add tip if not already added
        span_positions.append((x_edges[-1] - x_edges[0]) * pixel_to_m)
        chord_lengths.append(0.0)
        leading_edges.append(np.nan)
        trailing_edges.append(np.nan)
        section_centers.append(int(0.5 * (x_edges[-2] + x_edges[-1])))

    # --- Check lengths before creating DataFrame ---
    lengths = {
        "chord_lengths": len(chord_lengths),
        "leading_edges": len(leading_edges),
        "trailing_edges": len(trailing_edges),
        "section_centers": len(section_centers),
        "pitch_axis_m": 11
    }
    print("✅ Section data lengths:", lengths)

    if len(set(lengths.values())) > 1:
        print("❌ Mismatched lengths detected. Aborting.")
        return pd.DataFrame()

    # --- 5. Draw outline & symmetry ---
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    cv2.line(overlay, (symmetry_axis_px, 0), (symmetry_axis_px, h_img), (255, 255, 0), 2)

    # Draw magenta dots on the chosen pitching axis
    for cx in section_centers:
        cv2.circle(overlay, (cx, int(pitch_axis_y)), 3, (255, 0, 255), -1)

    # --- 7. Compute area and AR ---
    strip_width = (abs(x_tip - root_chord_x) / 11) * pixel_to_m
    area = np.sum([c * strip_width for c in chord_lengths])
    semi_span = 11 * strip_width
    AR = (4 * semi_span**2) / (2 * area) if area > 0 else np.nan
    mean_chord = np.mean(chord_lengths[:-1])  # exclude tip

    summary_metrics = {
        "half_area_m2": area,
        "semi_span_m": semi_span,
        "aspect_ratio": AR,
        "mean_chord_m": mean_chord
    }

    df = pd.DataFrame({
        "station": list(range(1, 12)),
        "leading_edge": leading_edges,
        "trailing_edge": trailing_edges,
        "chord_length": chord_lengths,
        "pitch_axis_m": [pitch_axis_m] * 11
    })
    
    df.to_csv(output_csv_path, index=False)

    # --- 5. Draw outline & symmetry ---
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    cv2.line(overlay, (symmetry_axis_px, 0), (symmetry_axis_px, h_img), (255, 255, 0), 2)

    # Draw magenta dots on the chosen pitching axis
    for cx in section_centers:
        cv2.circle(overlay, (cx, int(pitch_axis_y)), 3, (255, 0, 255), -1)

    # --- 7. Compute area and AR ---
    strip_width = (abs(x_tip - root_chord_x) / 11) * pixel_to_m
    area = np.sum([c * strip_width for c in chord_lengths])
    semi_span = 11 * strip_width
    AR = (4 * semi_span**2) / (2 * area) if area > 0 else np.nan
    mean_chord = np.mean(chord_lengths[:-1])  # exclude tip

    df = pd.DataFrame({
        "station": list(range(1, 12)),
        "leading_edge": leading_edges,
        "trailing_edge": trailing_edges,
        "chord_length": chord_lengths,
        "pitch_axis_m": [pitch_axis_m] * 11
    })
    
    df.to_csv(output_csv_path, index=False)

    # --- Draw final overlay with mirrored section lines ---
    # Draw outline and symmetry axis again
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    cv2.line(overlay, (symmetry_axis_px, 0), (symmetry_axis_px, h_img), (255, 255, 0), 2)

    # --- Draw original and mirrored section lines + pitch dots with correct station numbers ---
    for i in range(10):
        station_number = 10 - i
        station_number_mirror = i + 1

        cx = section_centers[i]
        y1 = int(leading_edges[i] / pixel_to_m)
        y2 = int(trailing_edges[i] / pixel_to_m)

        # Draw original section
        cv2.line(overlay, (cx, y1), (cx, y2), (0, 0, 255), 2)
        cv2.putText(overlay, f"{station_number}", (cx + 5, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.circle(overlay, (cx, int(pitch_axis_y)), 3, (255, 0, 255), -1)

        # Draw mirrored section
        mirrored_cx = int(2 * symmetry_axis_px - cx)
        cv2.line(overlay, (mirrored_cx, y1), (mirrored_cx, y2), (0, 0, 255), 2)
        cv2.putText(overlay, f"{11 - station_number_mirror}", (mirrored_cx + 5, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.circle(overlay, (mirrored_cx, int(pitch_axis_y)), 3, (255, 0, 255), -1)

    # Draw pitching axis across entire fluke
    cv2.line(overlay, (0, int(pitch_axis_y)), (overlay.shape[1], int(pitch_axis_y)), (255, 0, 255), 1)

    # --- Save overlay now ---
    cv2.imwrite(output_img_path, overlay)

    # --- Preview and retry-confirm loop ---
    while True:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Final mirrored fluke overlay – confirm to save or retry")

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

        if final_confirmed['value']:
            print(f"✅ Saved: {output_csv_path}\n🖼️ Overlay saved: {output_img_path}")
            return df, summary_metrics
        else:
            df, overlay = select_fluke_tip(
                img, refined_mask, pitch_axis_y,
                adjusted_axis_x, root_chord_x, scale_factor, pixel_to_m
            )

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
    output_img = os.path.join(output_folder, "fluke_overlay.png") 
    df, metrics = process_fluke_image_kmeans(cropped_img, output_csv, output_img)
    print("🎉 Done!")

# --- Run it ---
run_fluke_extraction_for_uav21_196e("/Users/georgesato/PhD/Chapter1/Fluke_Measurements")