import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
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

def select_fluke_tip(img, refined_mask, adjusted_axis_x, root_chord_x, scale_factor, datum_y):
    overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(refined_mask, cv2.COLOR_BGR2RGB), 0.3, 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.axhline(datum_y, color='lime', linestyle='--')
    ax.axvline(adjusted_axis_x, color='cyan', linestyle='--')
    ax.axvline(root_chord_x * scale_factor, color='orange', linestyle='--')
    ax.set_title("Step 4: Click fluke tip")

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='cyan', linestyle='--', label='Symmetry Axis'),
        Line2D([0], [0], color='orange', linestyle='--', label='Root Chord'),
        Line2D([0], [0], color='lime', linestyle='--', label='Datum')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    tip_point = plt.ginput(1, timeout=0)
    plt.close()

    if not tip_point:
        print("❌ No tip selected. Aborting.")
        return pd.DataFrame(), None, None

    (x_tip, y_tip) = tip_point[0]
    x_tip /= scale_factor
    y_tip /= scale_factor

    if x_tip < root_chord_x:
        root_chord_x, x_tip = x_tip, root_chord_x

    x_edges = np.linspace(x_tip, root_chord_x, 11)
    chord_lengths = []
    span_positions = []
    leading_edges = []
    trailing_edges = []
    section_end = []
    section_start = []
    le_datum_dists = []
    te_datum_dists = []

    # Create x_start and x_end arrays
    x_start = np.array(x_edges[1:])
    x_end = np.array(x_edges[:-1])   

    for i in range(10):
        section = refined_mask[:, int(x_end[i])]
        y_coords = np.where(section > 0)[0]
        print(y_coords)

        if len(y_coords) > 0:
            y_min = int(np.min(y_coords))
            y_max = int(np.max(y_coords))

            # Safely clamp y_min and y_max inside the actual image bounds
            y_min = np.clip(y_min, 0, overlay.shape[0]-1)
            y_max = np.clip(y_max, 0, overlay.shape[0]-1)

            le_from_datum = datum_y - y_min
            te_from_datum = datum_y - y_max
            chord = le_from_datum - te_from_datum  # same as y_max - y_min

            # Append distances for CSV
            le_datum_dists.append(round(le_from_datum,2))
            te_datum_dists.append(round(te_from_datum,2))

            # For plotting
            leading_edges.append(y_min)
            trailing_edges.append(y_max)
            chord_lengths.append(chord)
            section_start.append(x_start[i])
            section_end.append(x_end[i])
            span_positions.append(x_start[i] - x_edges[0])

    # Add tip if not already added    
    leading_edges.append(round(datum_y - y_tip,3))
    trailing_edges.append(round(datum_y - y_tip,3))
    le_datum_dists.append(round(datum_y - y_tip,3))
    te_datum_dists.append(round(datum_y - y_tip,3))
    chord_lengths.append(0.0)
    section_start.append(round(x_tip,3))
    section_end.append(round(x_tip,3))

    print(x_edges)
    print(x_start, x_end)
    print(span_positions)
    print(leading_edges)
    print(trailing_edges)
    print(chord_lengths)

    # b/chord ratios from Table 2 for stations 1–10
    b_over_c = [0.5, 0.456, 0.417, 0.383, 0.333, 0.265, 0.155, 0.0, -0.296, -0.875]
    pitch_axis_b_px = [round(b_over_c[i] * chord_lengths[i],3) if i < len(b_over_c) and chord_lengths[i] > 0 else np.nan for i in range(10)]
    pitch_axis_b_px.append(np.nan)

    print(pitch_axis_b_px)

    lengths = {
        "chord_lengths": len(np.array(chord_lengths)),
        "leading_edges": len(np.array(leading_edges)),
        "trailing_edges": len(np.array(trailing_edges)),
        "le_from_datum": len(np.array(le_datum_dists)),
        "te_from_datum": len(np.array(te_datum_dists)),
        "section_start": len(np.array(section_start)),
        "section_start_px": len(np.array(section_start)),
        "pitch_axis_b_px": len(np.array(pitch_axis_b_px))
    }
    print("✅ Section data lengths:", lengths)

    if len(set(lengths.values())) > 1:
        print("❌ Mismatched lengths detected. Aborting.")
        return pd.DataFrame(), None, None

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    h_img = img.shape[0]
    symmetry_axis_px = int(adjusted_axis_x)
    cv2.line(overlay, (symmetry_axis_px, 0), (symmetry_axis_px, h_img), (255, 255, 0), 2)
    cv2.line(overlay, (0, int(datum_y)), (overlay.shape[1], int(datum_y)), (0, 255, 0), 1)
    cv2.line(overlay, (int(root_chord_x), 0), (int(root_chord_x), h_img), (0, 165, 255), 1)

    for i in range(10):
        cx = int(section_end[i])
        b = pitch_axis_b_px[i]
        if not np.isnan(b):
            cy = int(leading_edges[i] + b)
            cv2.circle(overlay, (cx, cy), 3, (255, 0, 255), -1)

    strip_width_px = abs(x_tip - root_chord_x) / 11
    area_px2 = np.sum([c * strip_width_px for c in chord_lengths]) if chord_lengths else 0
    semi_span_px = 11 * strip_width_px
    AR_px = (4 * semi_span_px**2) / (2 * area_px2) if area_px2 > 0 else np.nan

    summary_metrics = {
        "strip_width_px": strip_width_px,
        "half_area_px2": area_px2,
        "semi_span_px": semi_span_px,
        "aspect_ratio": AR_px
    }

    df = pd.DataFrame({
        "station": list(range(1, 12)),
        "leading_edge_px": leading_edges,
        "trailing_edge_px": trailing_edges,
        "le_from_datum_px": le_datum_dists,
        "te_from_datum_px": te_datum_dists,
        "chord_length_px": chord_lengths,
        "section_start_px": section_start,
        "section_end_px": section_end,
        "pitch_axis_b_px": pitch_axis_b_px
    })

    return df, overlay, summary_metrics

def process_fluke_image(img, output_csv_path, output_img_path):
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

    # --- 2: Ask user to adjust root chord position (vertical base line) ---
    fig, ax = plt.subplots()
    fig.suptitle("Step 2: Adjust Root Chord (Vertical Base Line)", fontsize=14, weight='bold')
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(blended)
    default_root_x = display_rgb.shape[1] * 0.25
    root_line = ax.axvline(default_root_x, color='orange', linestyle='--', label='Root Chord')
    ax.axvline(adjusted_axis_x, color='cyan', linestyle='--')
    
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

    # --- 3: Add DATUM adjustment line (used for LE/TE distances) ---
    fig, ax = plt.subplots()
    fig.suptitle("Step 4: Adjust Horizontal Datum Line", fontsize=14, weight='bold')
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(blended)
    datum_y_default = display_rgb.shape[0] // 2
    datum_line = ax.axhline(datum_y_default, color='yellow', linestyle='--', linewidth=1.5, label='Datum')
    ax.axvline(adjusted_axis_x, color='cyan', linestyle='--')
    ax.axvline(root_chord_x * scale_factor, color='orange', linestyle='--')
    ax.legend(loc='upper right')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='cyan', linestyle='--', label='Symmetry Axis'),
        Line2D([0], [0], color='orange', linestyle='--', label='Root Chord'),
        Line2D([0], [0], color='yellow', linestyle='--', label='Datum') 
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax_slider_datum = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider_datum = Slider(ax_slider_datum, 'Datum Y', 0, display_rgb.shape[0], valinit=datum_y_default)

    datum_confirmed = {'value': False}

    def update_datum(val):
        datum_line.set_ydata([val, val])
        fig.canvas.draw_idle()

    slider_datum.on_changed(update_datum)

    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    btn_confirm_datum = Button(ax_button, 'Confirm')

    def confirm_datum(event):
        datum_confirmed['value'] = True
        plt.close()

    btn_confirm_datum.on_clicked(confirm_datum)
    plt.show()

    if not datum_confirmed['value']:
        return pd.DataFrame()

    datum_y = slider_datum.val

    # --- 4: Add horizontal cutoff to mask (removing peduncle above line) ---
    fig, ax = plt.subplots()
    fig.suptitle("Step 5: Adjust Cutoff Line (Removes Peduncle)", fontsize=14, weight='bold')
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(blended)
    cutoff_line = ax.axhline(display_rgb.shape[0] * 0.2, color='red', linestyle='--')
    ax.axvline(adjusted_axis_x, color='cyan', linestyle='--')
    ax.axvline(root_chord_x * scale_factor, color='orange', linestyle='--')
    ax.axhline(datum_y, color='yellow', linestyle='--')

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

    # --- 5: Click fluke tip ---
    df, overlay, summary_metrics = select_fluke_tip(img, refined_mask, adjusted_axis_x, root_chord_x, scale_factor, datum_y)

    # --- Draw final overlay with mirrored section lines ---
    # Draw outline and symmetry axis again
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    cv2.line(overlay, (symmetry_axis_px, 0), (symmetry_axis_px, h_img), (255, 255, 0), 2)

    # --- Draw original and mirrored section lines + pitch dots with correct station numbers ---
    for i in range(10):
        station_number = i + 1
        station_number_mirror = 10 - i

        cx = int(df.section_end_px[i])
        y1 = int(df.leading_edge_px[i])
        y2 = int(df.trailing_edge_px[i])

        # Draw original section
        cv2.line(overlay, (cx, y1), (cx, y2), (0, 0, 255), 2)
        cv2.putText(overlay, f"{station_number}", (cx, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw mirrored section
        mirrored_cx = int(2 * symmetry_axis_px - cx)
        cv2.line(overlay, (mirrored_cx, y1), (mirrored_cx, y2), (0, 0, 255), 2)
        cv2.putText(overlay, f"{11 - station_number_mirror}", (mirrored_cx - 5, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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

        # Add legend
        legend_elements = [
            Line2D([0], [0], color='cyan', linestyle='--', label='Symmetry Axis'),
            Line2D([0], [0], color='orange', linestyle='--', label='Root Chord'),
            Line2D([0], [0], color='yellow', linestyle='--', label='Datum'),
            Line2D([0], [0], marker='o', color='magenta', linestyle='None', label='Pitch Axis (b/chord)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        btn_confirm = Button(confirm_ax, 'Confirm')
        btn_retry = Button(retry_ax, 'Retry')
        btn_confirm.on_clicked(confirm_overlay)
        btn_retry.on_clicked(retry_overlay)

        plt.show()

        if final_confirmed['value']:
            df.to_csv(output_csv_path, index=False) 
            print(f"✅ Saved: {output_csv_path}\n🖼️ Overlay saved: {output_img_path}")
            return df, summary_metrics
        else:
            df, overlay = select_fluke_tip(
                img, refined_mask,
                adjusted_axis_x, root_chord_x, scale_factor, datum_y
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
    df, metrics = process_fluke_image(cropped_img, output_csv, output_img)
    print("🎉 Done!")

# --- Run it ---
run_fluke_extraction_for_uav21_196e("/Users/georgesato/PhD/Chapter1/Fluke_Measurements")