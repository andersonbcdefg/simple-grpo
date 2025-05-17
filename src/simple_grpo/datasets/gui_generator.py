from PIL import Image, ImageDraw, ImageFont
import random
import json
import math
import os

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Aesthetic & Sizing Adjustments
DEFAULT_FONT_SIZE = 11 # Slightly larger
CONTROL_BUTTON_RADIUS = 6 # Slightly larger controls
CONTROL_BUTTON_SPACING = 4
INTERNAL_BUTTON_WIDTH = 55 # Larger buttons
INTERNAL_BUTTON_HEIGHT = 25
INTERNAL_BUTTON_CORNER_RADIUS = 6
INTERNAL_BUTTON_MARGIN = 8
WINDOW_MARGIN = 10
MIN_WINDOW_SCALE_FACTOR = 0.4 # Window should be at least 40% of canvas size
TITLE_BAR_HEIGHT = 22 # More defined title bar
MIN_SHAPE_SIZE = 15
MAX_SHAPE_SIZE = 35
SHAPE_MARGIN = 5

# Colors (Normal Mode - macOS inspired)
WINDOW_BG_COLOR = "#F0F0F0"
WINDOW_TITLE_BAR_COLOR = "#E0E0E0" 
BUTTON_START_COLOR = "#6CBFEF" 
BUTTON_STOP_COLOR = "#FF7F7F" 
TEXT_COLOR = "#333333" 
CONTROL_RED = "#FF5F57"
CONTROL_YELLOW = "#FFBD2E"
CONTROL_GREEN = "#27C93F"
CONTROL_RED_OUTLINE = "#E0443E"
CONTROL_YELLOW_OUTLINE = "#DEA123"
CONTROL_GREEN_OUTLINE = "#1AAB29"
SHAPE_COLORS = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E0BBE4"]

# Variations for Hard Mode
HARD_WINDOW_BG_COLORS = ["#3C3C3C", "#2E4053", "#F5F5DC"] # Dark grey, dark blue, beige
HARD_TITLE_BAR_COLORS = ["#505050", "#1F2D3A", "#D2B48C"] # Matching darker/different title bars
HARD_BUTTON_TEXTS = {
    "Start": ["Go", "Begin", "Run", "Execute"],
    "Stop": ["Halt", "End", "Cancel", "Abort"]
}
HARD_BUTTON_BG_COLORS = ["#FF8C00", "#9370DB", "#3CB371", "#D2691E"] # DarkOrange, MediumPurple, MediumSeaGreen, Chocolate
HARD_TEXT_COLORS = ["#FFFFFF", "#F0F0F0", "#000000"]
HARD_SHAPE_COLORS = ["#DC143C", "#00CED1", "#FFD700", "#00FA9A", "#FF69B4", "#8A2BE2"] # More vibrant/different colors


class GUIElement:
    """Represents a single element in the GUI scene."""
    def __init__(self, name: str, center_x: int, center_y: int, bounding_box: tuple[int, int, int, int]):
        self.name = name
        self.center_x = center_x
        self.center_y = center_y
        self.bounding_box = bounding_box # (x_min, y_min, x_max, y_max)

    def to_dict(self):
        """Converts the element to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "bounding_box": self.bounding_box
        }

class GUIGenerator:
    """Generates synthetic GUI scenes with various elements."""

    def __init__(self, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, seed=None):
        self.width = width
        self.height = height
        self.elements: list[GUIElement] = []
        if seed is not None:
            random.seed(seed)
        
        # Improved font loading - try common sans-serif fonts
        font_names = ["DejaVuSans.ttf", "Helvetica.ttf", "Arial.ttf", "arial.ttf"]
        font_loaded = False
        for font_name in font_names:
            try:
                self.font = ImageFont.truetype(font_name, DEFAULT_FONT_SIZE)
                font_loaded = True
                break
            except IOError:
                continue
        if not font_loaded:
            try:
                self.font = ImageFont.load_default(size=DEFAULT_FONT_SIZE)
            except TypeError:
                self.font = ImageFont.load_default()


    def _add_element(self, name: str, bbox: tuple[int, int, int, int]):
        """Helper to create and add a GUIElement."""
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        self.elements.append(GUIElement(name, center_x, center_y, bbox))

    def _draw_window_and_controls(self, draw: ImageDraw.ImageDraw, hard_mode: bool = False) -> tuple[int, int, int, int] | None:
        """Draws the main window and its control buttons. Returns window content bounding box."""
        min_w = int(self.width * MIN_WINDOW_SCALE_FACTOR)
        min_h = int(self.height * MIN_WINDOW_SCALE_FACTOR)

        w_width = random.randint(min_w, self.width - 2 * WINDOW_MARGIN)
        w_height = random.randint(min_h, self.height - 2 * WINDOW_MARGIN)
        
        # Ensure minimum size includes title bar
        if w_height < TITLE_BAR_HEIGHT + INTERNAL_BUTTON_HEIGHT + 2 * INTERNAL_BUTTON_MARGIN:
            w_height = TITLE_BAR_HEIGHT + INTERNAL_BUTTON_HEIGHT + 2 * INTERNAL_BUTTON_MARGIN

        w_x1 = random.randint(WINDOW_MARGIN, self.width - w_width - WINDOW_MARGIN)
        w_y1 = random.randint(WINDOW_MARGIN, self.height - w_height - WINDOW_MARGIN)
        w_x2 = w_x1 + w_width
        w_y2 = w_y1 + w_height
        window_bbox = (w_x1, w_y1, w_x2, w_y2)

        # Draw Title Bar
        title_bar_bbox = (w_x1, w_y1, w_x2, w_y1 + TITLE_BAR_HEIGHT)
        draw.rectangle(title_bar_bbox, fill=WINDOW_TITLE_BAR_COLOR)
        # Draw Window Body
        window_body_bbox = (w_x1, w_y1 + TITLE_BAR_HEIGHT, w_x2, w_y2)
        draw.rectangle(window_body_bbox, fill=WINDOW_BG_COLOR, outline="#B0B0B0") # Slightly darker outline for body
        self._add_element("window", window_bbox) # Overall window bounding box

        # Window controls (on title bar, left-aligned for macOS style)
        control_y_center = w_y1 + TITLE_BAR_HEIGHT // 2
        
        red_x_center = w_x1 + CONTROL_BUTTON_SPACING + CONTROL_BUTTON_RADIUS
        red_bbox = (red_x_center - CONTROL_BUTTON_RADIUS, control_y_center - CONTROL_BUTTON_RADIUS, 
                    red_x_center + CONTROL_BUTTON_RADIUS, control_y_center + CONTROL_BUTTON_RADIUS)
        draw.ellipse(red_bbox, fill="#FF5F57", outline="#E0443E") # macOS Red
        self._add_element("window_close_button", red_bbox)

        yellow_x_center = red_x_center + CONTROL_BUTTON_SPACING + 2 * CONTROL_BUTTON_RADIUS
        yellow_bbox = (yellow_x_center - CONTROL_BUTTON_RADIUS, control_y_center - CONTROL_BUTTON_RADIUS, 
                       yellow_x_center + CONTROL_BUTTON_RADIUS, control_y_center + CONTROL_BUTTON_RADIUS)
        draw.ellipse(yellow_bbox, fill="#FFBD2E", outline="#DEA123") # macOS Yellow
        self._add_element("window_minimize_button", yellow_bbox)
        
        green_x_center = yellow_x_center + CONTROL_BUTTON_SPACING + 2 * CONTROL_BUTTON_RADIUS
        green_bbox = (green_x_center - CONTROL_BUTTON_RADIUS, control_y_center - CONTROL_BUTTON_RADIUS, 
                      green_x_center + CONTROL_BUTTON_RADIUS, control_y_center + CONTROL_BUTTON_RADIUS)
        draw.ellipse(green_bbox, fill="#27C93F", outline="#1AAB29") # macOS Green
        self._add_element("window_maximize_button", green_bbox)
        
        # Define content area (within window_body_bbox)
        content_x1 = w_x1 + 1 
        content_y1 = w_y1 + TITLE_BAR_HEIGHT + 1
        content_x2 = w_x2 - 1
        content_y2 = w_y2 - 1
        
        if content_y1 >= content_y2 - INTERNAL_BUTTON_MARGIN or content_x1 >= content_x2 - INTERNAL_BUTTON_MARGIN: 
            return None 
        return (content_x1, content_y1, content_x2, content_y2)

    def _draw_internal_buttons(self, draw: ImageDraw.ImageDraw, content_bbox: tuple[int, int, int, int], hard_mode: bool = False):
        """Draws 'start' and 'stop' buttons within the window's content area."""
        con_x1, con_y1, con_x2, con_y2 = content_bbox

        min_content_width = INTERNAL_BUTTON_WIDTH + 2 * INTERNAL_BUTTON_MARGIN
        min_content_height = INTERNAL_BUTTON_HEIGHT + 2 * INTERNAL_BUTTON_MARGIN
        if (con_x2 - con_x1) < min_content_width or \
           (con_y2 - con_y1) < min_content_height:
            return 

        def place_button(name: str, button_text_color: str, button_bg_color: str, existing_bboxes: list):
            max_attempts = 20
            for _ in range(max_attempts):
                # Ensure button fits
                btn_x1 = random.randint(con_x1 + INTERNAL_BUTTON_MARGIN, max(con_x1 + INTERNAL_BUTTON_MARGIN, con_x2 - INTERNAL_BUTTON_WIDTH - INTERNAL_BUTTON_MARGIN))
                btn_y1 = random.randint(con_y1 + INTERNAL_BUTTON_MARGIN, max(con_y1 + INTERNAL_BUTTON_MARGIN, con_y2 - INTERNAL_BUTTON_HEIGHT - INTERNAL_BUTTON_MARGIN))
                btn_x2 = btn_x1 + INTERNAL_BUTTON_WIDTH
                btn_y2 = btn_y1 + INTERNAL_BUTTON_HEIGHT
                current_bbox = (btn_x1, btn_y1, btn_x2, btn_y2)

                # Check if button is fully within content_bbox
                if not (btn_x1 >= con_x1 and btn_y1 >= con_y1 and btn_x2 <= con_x2 and btn_y2 <= con_y2):
                    continue
                    
                is_overlapping = False
                for eb_x1, eb_y1, eb_x2, eb_y2 in existing_bboxes:
                    if not (btn_x2 < eb_x1 or btn_x1 > eb_x2 or btn_y2 < eb_y1 or btn_y1 > eb_y2):
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    # Try to use rounded_rectangle if available (Pillow 9.0.0+)
                    if hasattr(draw, "rounded_rectangle"):
                        draw.rounded_rectangle(current_bbox, radius=INTERNAL_BUTTON_CORNER_RADIUS, fill=button_bg_color, outline="#ADADAD")
                    else:
                        draw.rectangle(current_bbox, fill=button_bg_color, outline="#ADADAD")
                    
                    try:
                        text_bbox = draw.textbbox((0,0), name, font=self.font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError: 
                        text_width = int(len(name) * (self.font.size if hasattr(self.font, 'size') else DEFAULT_FONT_SIZE * 0.6))
                        text_height = int(self.font.size if hasattr(self.font, 'size') else DEFAULT_FONT_SIZE)

                    text_x = btn_x1 + (INTERNAL_BUTTON_WIDTH - text_width) / 2
                    text_y = btn_y1 + (INTERNAL_BUTTON_HEIGHT - text_height) / 2 -1 # Small adjustment for better vertical centering
                    draw.text((text_x, text_y), name, fill=button_text_color, font=self.font)
                    # IMPORTANT: Use the *original semantic name* for the element, regardless of displayed text
                    semantic_name = "start_button" if name.lower() in [t.lower() for t in ["Start"] + HARD_BUTTON_TEXTS["Start"]] else "stop_button"
                    self._add_element(semantic_name, current_bbox)
                    existing_bboxes.append(current_bbox)
                    return True
            return False

        placed_button_bboxes = []
        # Determine text and colors based on mode
        if hard_mode:
            start_text = random.choice(HARD_BUTTON_TEXTS["Start"])
            stop_text = random.choice(HARD_BUTTON_TEXTS["Stop"])
            start_bg = random.choice(HARD_BUTTON_BG_COLORS)
            stop_bg = random.choice(HARD_BUTTON_BG_COLORS)
            text_color = random.choice(HARD_TEXT_COLORS)
        else:
            start_text = "Start"
            stop_text = "Stop"
            start_bg = BUTTON_START_COLOR
            stop_bg = BUTTON_STOP_COLOR
            text_color = TEXT_COLOR

        place_button(start_text, text_color, start_bg, placed_button_bboxes)
        
        # Heuristic to check if there's space for a second button
        if (con_x2 - con_x1) >= (INTERNAL_BUTTON_WIDTH * 2 + INTERNAL_BUTTON_MARGIN * 3) or \
           (con_y2 - con_y1) >= (INTERNAL_BUTTON_HEIGHT * 2 + INTERNAL_BUTTON_MARGIN * 3): 
            place_button(stop_text, text_color, stop_bg, placed_button_bboxes)


    def _draw_random_shapes(self, draw: ImageDraw.ImageDraw, num_shapes: int, hard_mode: bool = False):
        """Draws a number of random simple shapes on the canvas."""
        shape_types = ["rectangle", "ellipse"]
        colors = HARD_SHAPE_COLORS if hard_mode else SHAPE_COLORS
        outline_color = "#555555" if hard_mode else "#BFBFBF"

        for i in range(num_shapes):
            shape_type = random.choice(shape_types)
            size_w = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
            size_h = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
            
            x1 = random.randint(SHAPE_MARGIN, self.width - size_w - SHAPE_MARGIN)
            y1 = random.randint(SHAPE_MARGIN, self.height - size_h - SHAPE_MARGIN)
            x2 = x1 + size_w
            y2 = y1 + size_h
            
            bbox = (x1, y1, x2, y2)
            color = random.choice(colors)
            element_name = f"random_{shape_type}_{i+1}"

            if shape_type == "rectangle":
                draw.rectangle(bbox, fill=color, outline=outline_color)
            elif shape_type == "ellipse":
                draw.ellipse(bbox, fill=color, outline=outline_color)
            
            self._add_element(element_name, bbox)

    def generate_scene(self, hard_mode: bool = False) -> tuple[Image.Image, str]:
        """Generates a new GUI scene image and corresponding JSON data for all elements."""
        self.elements = [] 
        desktop_bg = "#303030" if hard_mode else "#EAEAEA"
        image = Image.new("RGB", (self.width, self.height), desktop_bg) 
        draw = ImageDraw.Draw(image)

        window_content_bbox = self._draw_window_and_controls(draw, hard_mode=hard_mode)
        
        if window_content_bbox:
            self._draw_internal_buttons(draw, window_content_bbox, hard_mode=hard_mode)

        self._draw_random_shapes(draw, num_shapes=random.randint(1, 3), hard_mode=hard_mode)

        elements_json = json.dumps([element.to_dict() for element in self.elements], indent=2)
        return image, elements_json

    def generate_scene_with_target(self, generate_hard_mode: bool = False) -> tuple[Image.Image, str, dict | None]:
        """
        Generates a GUI scene (potentially hard mode), selects a random interactive target element,
        and returns the image, JSON of all elements, and the target element's details.
        """
        image, all_elements_json = self.generate_scene(hard_mode=generate_hard_mode) 

        interactive_element_names = [
            "window_close_button", "window_minimize_button", "window_maximize_button",
            "start_button", "stop_button"
        ]
        
        candidate_targets = []
        for elem in self.elements:
            if elem.name in interactive_element_names:
                candidate_targets.append(elem.to_dict())
        
        if not candidate_targets:
            return image, all_elements_json, None
            
        selected_target = random.choice(candidate_targets)
        return image, all_elements_json, selected_target

    @staticmethod
    def plot_predictions(image: Image.Image, predictions_data: str | list, pred_color="#FF00FF", truth_color="#00FF00", default_font_size=9, x_marker_size=4) -> Image.Image:
        """
        Plots predicted and/or ground truth annotations on an image.
        predictions_data: JSON string or list of dicts.
        Elements in predictions_data can have an optional 'is_truth': True key to be styled with truth_color.
        """
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        img_width, img_height = output_image.size # Get image dimensions
        
        try:
            if isinstance(predictions_data, str):
                predictions = json.loads(predictions_data)
            elif isinstance(predictions_data, list):
                predictions = predictions_data
            else:
                print(f"Warning: Invalid type for predictions_data: {type(predictions_data)}. Expected str or list.")
                predictions = []
        except json.JSONDecodeError:
            print(f"Error decoding predictions JSON: {predictions_data[:100]}...")
            predictions = []

        # Font for labels on plot
        plot_font = None
        font_names = ["DejaVuSans.ttf", "Helvetica.ttf", "Arial.ttf", "arial.ttf"]
        for font_name in font_names:
            try:
                plot_font = ImageFont.truetype(font_name, default_font_size)
                break
            except IOError:
                continue
        if not plot_font:
            try:
                plot_font = ImageFont.load_default(size=default_font_size)
            except TypeError:
                plot_font = ImageFont.load_default()


        for pred_elem in predictions:
            if not isinstance(pred_elem, dict): 
                print(f"Warning: Skipping non-dict element in predictions list: {pred_elem}")
                continue

            name = pred_elem.get("name", "unknown")
            bbox = pred_elem.get("bounding_box")
            center_x = pred_elem.get("center_x")
            center_y = pred_elem.get("center_y")

            current_color = truth_color if pred_elem.get("is_truth") else pred_color
            
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                pred_x1, pred_y1, pred_x2, pred_y2 = bbox
                draw.rectangle((pred_x1, pred_y1, pred_x2, pred_y2), outline=current_color, width=2) # Thicker outline for bbox
                
                try:
                    text_bbox_plot = plot_font.getbbox(name) # Pillow 9+
                    text_height = text_bbox_plot[3] - text_bbox_plot[1]
                except AttributeError: 
                    text_height = default_font_size 
                
                label_y = pred_y1 - text_height - 2 if pred_y1 - text_height -2 > 0 else pred_y1 + 2
                draw.text((pred_x1 + 2, label_y), name, fill=current_color, font=plot_font)

            elif center_x is not None and center_y is not None:
                # Clamp coordinates before drawing the 'X' marker
                plot_x = max(0, min(img_width - 1, int(center_x)))
                plot_y = max(0, min(img_height - 1, int(center_y)))
                
                # Draw an 'X' marker for point predictions at the (potentially clamped) coordinates
                draw.line([(plot_x - x_marker_size, plot_y - x_marker_size),
                           (plot_x + x_marker_size, plot_y + x_marker_size)],
                          fill=current_color, width=2)
                draw.line([(plot_x - x_marker_size, plot_y + x_marker_size),
                           (plot_x + x_marker_size, plot_y - x_marker_size)],
                          fill=current_color, width=2)
                
                try:
                    text_bbox_plot = plot_font.getbbox(name)
                    text_height = text_bbox_plot[3] - text_bbox_plot[1]
                except AttributeError: 
                    text_height = default_font_size
                
                label_y_offset = text_height + x_marker_size + 2
                label_y = plot_y - label_y_offset if plot_y - label_y_offset > 0 else plot_y + x_marker_size + 2
                draw.text((plot_x + x_marker_size + 2, label_y), name, fill=current_color, font=plot_font)
        return output_image

if __name__ == '__main__':

    # Original main block for generating polished examples and comparisons:
    generator_main_example = GUIGenerator(seed=42)
    generated_image_main, elements_json_data_main = generator_main_example.generate_scene()
    generated_image_main.save("gui_scene_polished_example.png")
    with open("gui_scene_polished_elements.json", "w") as f:
        f.write(elements_json_data_main)
    print("Generated: gui_scene_polished_example.png, gui_scene_polished_elements.json")

    gt_elements_list = json.loads(elements_json_data_main)
    for elem in gt_elements_list:
        elem["is_truth"] = True
    image_with_ground_truth = GUIGenerator.plot_predictions(generated_image_main, gt_elements_list)
    image_with_ground_truth.save("gui_scene_polished_with_ground_truth.png")
    print("Generated: gui_scene_polished_with_ground_truth.png")

    mock_predictions_list = []
    original_elements = json.loads(elements_json_data_main)
    if original_elements:
        for i, elem_data in enumerate(original_elements):
            pred_name = elem_data["name"]
            if "window_minimize_button" in pred_name: 
                continue
            if i % 2 == 0 and "random" not in pred_name: 
                mock_predictions_list.append({
                    "name": "pred_" + pred_name + "_click",
                    "center_x": elem_data["center_x"] + random.randint(-4, 4),
                    "center_y": elem_data["center_y"] + random.randint(-4, 4),
                    "is_truth": False
                })
            else: 
                new_bbox = [b + random.randint(-7, 7) for b in elem_data["bounding_box"]]
                mock_predictions_list.append({
                    "name": "pred_" + pred_name,
                    "center_x": (new_bbox[0] + new_bbox[2]) // 2,
                    "center_y": (new_bbox[1] + new_bbox[3]) // 2,
                    "bounding_box": new_bbox,
                    "is_truth": False
                })
        mock_predictions_list.append({
            "name": "false_positive_click",
            "center_x": random.randint(20, IMAGE_WIDTH - 20),
            "center_y": random.randint(20, IMAGE_HEIGHT - 20),
            "is_truth": False
        })
    image_with_mock_predictions = GUIGenerator.plot_predictions(generated_image_main, mock_predictions_list)
    image_with_mock_predictions.save("gui_scene_polished_with_mock_predictions.png")
    print("Generated: gui_scene_polished_with_mock_predictions.png")

    combined_plot_data = gt_elements_list + mock_predictions_list
    comparison_image_with_both = GUIGenerator.plot_predictions(generated_image_main, combined_plot_data)
    comparison_image_with_both.save("gui_scene_polished_comparison.png")
    print("Generated: gui_scene_polished_comparison.png (green=truth, magenta=prediction)")

    # --- Generate 15 examples for variability check ---
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    print(f"\nGenerating 15 examples in '{examples_dir}/' directory...")
    example_generator_loop = GUIGenerator() 
    for i in range(15):
        example_image, _ = example_generator_loop.generate_scene() # Using original generate_scene for these examples
        example_image.save(os.path.join(examples_dir, f"gui_scene_example_{i+1:02d}.png"))
    print(f"Finished generating 15 examples in '{examples_dir}/'.") 

    # Add example for hard mode generation
    print("\n--- Generating Hard Mode Example ---")
    hard_generator = GUIGenerator(seed=123)
    hard_image, hard_json, hard_target = hard_generator.generate_scene_with_target(generate_hard_mode=True)
    hard_image.save("gui_scene_hard_mode_example.png")
    print("Saved: gui_scene_hard_mode_example.png")
    print("Hard mode JSON sample:")
    # print(hard_json)
    print("Hard mode target:")
    print(json.dumps(hard_target, indent=2) if hard_target else "None")

    # Plot target on hard mode image
    if hard_target:
         plot_data_hard = [{**hard_target, "is_truth": True, "name": "TARGET_"+hard_target["name"]}]
         img_hard_w_target = GUIGenerator.plot_predictions(hard_image, plot_data_hard, truth_color="cyan")
         img_hard_w_target.save("gui_scene_hard_mode_WITH_TARGET.png")
         print("Saved: gui_scene_hard_mode_WITH_TARGET.png") 