import base64
import cv2
import re
import copy
import io
import numpy as np
from PIL import Image
from colorama import Fore, Style

def print_with_color(text: str, color=""):
    if color == "red":
        print(Fore.RED + text)
    elif color == "green":
        print(Fore.GREEN + text)
    elif color == "yellow":
        print(Fore.YELLOW + text)
    elif color == "blue":
        print(Fore.BLUE + text)
    elif color == "magenta":
        print(Fore.MAGENTA + text)
    elif color == "cyan":
        print(Fore.CYAN + text)
    elif color == "white":
        print(Fore.WHITE + text)
    elif color == "black":
        print(Fore.BLACK + text)
    else:
        print(text)
    print(Style.RESET_ALL)

def encode_image(image_path):
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return base64_encoded_data

def encode_image_pillow(image_pillow):
    buffer = io.BytesIO()
    image_pillow.save(buffer, format="PNG")
    base64_encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return base64_encoded_data
    
def draw_grid(img_path, output_path, scale=1):
    def get_unit_len(n):
        for i in range(1, n + 1):
            if n % i == 0 and 120 <= i <= 180:
                return i
        return -1

    image = cv2.imread(img_path)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    height, width, _ = image.shape
    color = (255, 116, 113)
    unit_height = get_unit_len(height)
    if unit_height < 0:
        unit_height = 120
    unit_width = get_unit_len(width)
    if unit_width < 0:
        unit_width = 120
    thick = int(unit_width // 50)
    rows = height // unit_height
    cols = width // unit_width
    for i in range(rows):
        for j in range(cols):
            label = i * cols + j + 1
            left = int(j * unit_width)
            top = int(i * unit_height)
            right = int((j + 1) * unit_width)
            bottom = int((i + 1) * unit_height)
            cv2.rectangle(image, (left, top), (right, bottom), color, thick//2)
            cv2.putText(image, str(label), (left + int(unit_width * 0.05) + 3, top + int(unit_height * 0.25) + 3), 0, 1.2, (0, 0, 0), thick)
            cv2.putText(image, str(label), (left + int(unit_width * 0.05), top + int(unit_height * 0.25)), 0, 1.2, color, thick)
    cv2.imwrite(output_path, image)
    return rows, cols, height, width

def parse_grid_rsp_given_start(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0].replace("`","")
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")

        act_name = act.split("(")[0]
        if act_name == "set_1_point":
            params = re.findall(r"set_1_point\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            return [act_name, start_area, start_subarea], act
        elif act_name == "set_2_points":
            params = re.findall(r"set_2_points\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            end_area = int(params[2].strip())
            end_subarea = params[3].strip()[1:-1]
            return [act_name, start_area, start_subarea, end_area, end_subarea], act
        elif act_name == "set_3_points":
            params = re.findall(r"set_3_points\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            mid_area = int(params[2].strip())
            mid_subarea = params[3].strip()[1:-1]
            end_area = int(params[4].strip())
            end_subarea = params[5].strip()[1:-1]
            return [act_name, start_area, start_subarea, mid_area, mid_subarea, end_area, end_subarea], act
        elif act_name == "set_4_points":
            params = re.findall(r"set_4_points\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            mid_1_area = int(params[2].strip())
            mid_1_subarea = params[3].strip()[1:-1]
            mid_2_area = int(params[4].strip())
            mid_2_subarea = params[5].strip()[1:-1]
            end_area = int(params[6].strip())
            end_subarea = params[7].strip()[1:-1]
            return [act_name, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, end_area, end_subarea], act
        elif act_name == "set_5_points":
            params = re.findall(r"set_5_points\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            mid_1_area = int(params[2].strip())
            mid_1_subarea = params[3].strip()[1:-1]
            mid_2_area = int(params[4].strip())
            mid_2_subarea = params[5].strip()[1:-1]
            mid_3_area = int(params[6].strip())
            mid_3_subarea = params[7].strip()[1:-1]
            end_area = int(params[8].strip())
            end_subarea = params[9].strip()[1:-1]
            return [act_name, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, mid_3_area, mid_3_subarea, end_area, end_subarea], act
        elif act_name == "set_6_points":
            params = re.findall(r"set_6_points\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            mid_1_area = int(params[2].strip())
            mid_1_subarea = params[3].strip()[1:-1]
            mid_2_area = int(params[4].strip())
            mid_2_subarea = params[5].strip()[1:-1]
            mid_3_area = int(params[6].strip())
            mid_3_subarea = params[7].strip()[1:-1]
            mid_4_area = int(params[8].strip())
            mid_4_subarea = params[9].strip()[1:-1]
            end_area = int(params[10].strip())
            end_subarea = params[11].strip()[1:-1]
            return [act_name, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, mid_3_area, mid_3_subarea, mid_4_area, mid_4_subarea, end_area, end_subarea], act
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"], None
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"], None

def parse_grid_rsp_judge(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0].replace("`","")
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")

        act_name = act.split("(")[0]
        if act_name == "set_point":
            params = re.findall(r"set_point\((.*?)\)", act)[0].split(",")
            area = int(params[0].strip())
            subarea = params[1].strip()[1:-1]
            return [act_name, area, subarea]
        elif act_name == "judge_point":
            params = re.findall(r"judge_point\((.*?)\)", act)[0].split(",")
            judgement = eval(params[0].strip()[1:-1])
            area = int(params[1].strip())
            subarea = params[2].strip()[1:-1]
            return [act_name, judgement, area, subarea]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]

def parse_grid_rsp_select(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0].replace("`","")
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")

        act_name = act.split("(")[0]
        if act_name == "select_object":
            params = re.findall(r"select_object\((.*?)\)", act)[0].split(",")
            index = int(params[0].strip())
            return [act_name, index], act
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"], None
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"], None

def area_to_xy(area, subarea, rows, cols, height, width):
    area -= 1
    row, col = area // cols, area % cols
    x_0, y_0 = col * (width // cols), row * (height // rows)
    if subarea == "top-left":
        x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 4
    elif subarea == "top":
        x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 4
    elif subarea == "top-right":
        x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 4
    elif subarea == "left":
        x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 2
    elif subarea == "right":
        x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) // 2
    elif subarea == "bottom-left":
        x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) * 3 // 4
    elif subarea == "bottom":
        x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) * 3 // 4
    elif subarea == "bottom-right":
        x, y = x_0 + (width // cols) * 3 // 4, y_0 + (height // rows) * 3 // 4
    else:
        x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 2
    return x, y

def xy_to_area(x, y, rows, cols, height, width):
    col = x // (width // cols)
    row = y // (height // rows)
    area = int(row * cols + col + 1)  # 确保返回 int 类型
    
    x_0, y_0 = col * (width // cols), row * (height // rows)
    sub_x = x - x_0
    sub_y = y - y_0
    
    if sub_x < (width // cols) // 2 and sub_y < (height // rows) // 2:
        subarea = "top-left"
    elif sub_x >= (width // cols) // 2 and sub_y < (height // rows) // 2:
        subarea = "top-right"
    elif sub_x < (width // cols) // 2 and sub_y >= (height // rows) // 2:
        subarea = "bottom-left"
    elif sub_x >= (width // cols) // 2 and sub_y >= (height // rows) // 2:
        subarea = "bottom-right"
    elif sub_x < (width // cols) // 2:
        subarea = "left"
    elif sub_x >= (width // cols) // 2:
        subarea = "right"
    elif sub_y < (height // rows) // 2:
        subarea = "top"
    else:
        subarea = "bottom"

    return area, subarea

def plot_trajectory(trajectory, img_path_input, img_path_output):
    image = Image.open(img_path_input)
    transparent_background = copy.deepcopy(image).convert('RGBA')
    w, h = transparent_background.size
    
    transparent_layer = np.zeros((h, w, 4))
    
    if len(trajectory) > 1:
        color = (255, 255, 255, 255)
        thickness = 10
        for i in range(len(trajectory)-1):
            start_point = trajectory[i]
            end_point = trajectory[i+1]
            if i == len(trajectory)-2:
                cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), color, thickness, tipLength=0.1)
            else:
                cv2.line(transparent_layer, tuple(start_point), tuple(end_point), color, thickness)
    else:
        color = (113, 116, 255, 255)
        thickness = 20
        cv2.circle(transparent_layer, tuple(trajectory[0]), thickness, color, -1)
                
    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    
    trajectory_map.save(img_path_output)

def plot_trajectory_multi(trajectory_list, img_path_input, img_path_output):
    image = Image.open(img_path_input)
    transparent_background = copy.deepcopy(image).convert('RGBA')
    w, h = transparent_background.size
    
    transparent_layer = np.zeros((h, w, 4))
    
    for trajectory in trajectory_list:
        color = (255, 255, 255, 255)
        thickness = 10
        for i in range(len(trajectory)-1):
            start_point = trajectory[i]
            end_point = trajectory[i+1]
            if i == len(trajectory)-2:
                cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), color, thickness, tipLength=0.1)
            else:
                cv2.line(transparent_layer, tuple(start_point), tuple(end_point), color, thickness)
                
    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    
    trajectory_map.save(img_path_output)

def plot_point(points, img_path_input, img_path_output):
    start_point = points[0]
    end_point = points[1]
    
    image = Image.open(img_path_input)
    transparent_background = copy.deepcopy(image).convert('RGBA')
    w, h = transparent_background.size
    
    transparent_layer = np.zeros((h, w, 4))
    cv2.circle(transparent_layer, tuple(start_point), 15, (255, 255, 0, 255), -1)
    cv2.circle(transparent_layer, tuple(end_point), 15, (0, 255, 255, 255), -1)
    
    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    
    trajectory_map.save(img_path_output)

def parse_prompt_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        reply = re.findall(r"Reply: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Reply:", "yellow")
        print_with_color(reply, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        
        sentence_list = re.findall(r"<(.*?)>", reply, re.MULTILINE)
        
        if len(sentence_list) == 0:
            return ["ERROR"]

        return sentence_list
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]
    
def parse_camera_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0].replace("`","")
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")

        act_name = act.split("(")[0]
        
        if act_name == "set_camera_motion":
            params = re.findall(r"set_camera_motion\((.*?)\)", act)[0].split(",")
            x_translation = float(params[0].strip())
            y_translation = float(params[1].strip())
            z_translation = float(params[2].strip())
            x_rotation = float(params[3].strip())
            y_rotation = float(params[4].strip())
            z_rotation = float(params[5].strip())
            motion_type = params[6].strip()[1:-1]
            return [act_name, x_translation, y_translation, z_translation, x_rotation, y_rotation, z_rotation, motion_type], act
        elif act_name == "set_camera_motion_complex":
            params = re.findall(r"set_camera_motion_complex\((.*?)\)", act)[0].split(",")
            x_shift_1 = float(params[0].strip())
            x_shift_2 = float(params[1].strip())
            y_shift_1 = float(params[2].strip())
            y_shift_2 = float(params[3].strip())
            z_shift_1 = float(params[4].strip())
            z_shift_2 = float(params[5].strip())
            x_rotation = float(params[6].strip())
            y_rotation = float(params[7].strip())
            z_rotation = float(params[8].strip())
            return [act_name, x_shift_1, x_shift_2, y_shift_1, y_shift_2, z_shift_1, z_shift_2, x_rotation, y_rotation, z_rotation], act
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"], None
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"], None