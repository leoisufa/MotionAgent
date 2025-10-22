import os
import re
import time
import numpy as np
import agent.prompts as prompts
from agent.agent_utils import draw_grid, parse_grid_rsp_given_start, area_to_xy, xy_to_area, plot_trajectory, plot_trajectory_multi, parse_grid_rsp_judge, parse_grid_rsp_select

def agent_single_trajectory(image_path, save_path, task_desc, mllm, round_max=10):
    start_area, start_subarea = "None", "None"
    output_image_point = ""

    grid_image = os.path.join(save_path, "grid.png")
    rows, cols, height, width = draw_grid(image_path, grid_image)

    round_count = 0
    while round_count < round_max:
        round_count += 1
        
        if round_count == 1:
            prompt = prompts.task_template_set_start_point
            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            status, rsp = mllm.get_model_response(prompt, grid_image)
        else:
            prompt = prompts.task_template_judge_start_point
            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            point_location = f"({str(start_area)}, {start_subarea})"
            prompt = re.sub(r"<point_location>", point_location, prompt)
            status, rsp = mllm.get_model_response(prompt, output_image_point)
        
        if status:
            res = parse_grid_rsp_judge(rsp)
        
            if res[0] == "ERROR":
                round_count -= 1
                time.sleep(5)
                continue
            
            act_name = res[0]

            if act_name == "set_point":
                _, start_area, start_subarea = res
                x, y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                output_image_point = os.path.join(save_path, f"start_{round_count}.png")
                plot_trajectory([[x, y]], grid_image, output_image_point)
            elif act_name == "judge_point":
                _, judgement, start_area, start_subarea = res
                x, y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                output_image_point = os.path.join(save_path, f"start_{round_count}.png")
                plot_trajectory([[x, y]], grid_image, output_image_point)
                if judgement:
                    break
            else:
                round_count -= 1
                time.sleep(5)
                continue
        else:
            round_count -= 1
            time.sleep(5)
            continue
    
    trajectory = []
    round_count = 0
    while round_count < round_max:
        round_count += 1

        prompt = prompts.task_template_trajectory_given_start
        prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
        start_point_location = f"({str(start_area)}, {start_subarea})"
        prompt = re.sub(r"<start_point_location>", start_point_location, prompt)
        status, rsp = mllm.get_model_response(prompt, output_image_point)

        if status:
            res = parse_grid_rsp_given_start(rsp)
        
            if res[0] == "ERROR":
                round_count -= 1
                time.sleep(5)
                continue
            
            act_name = res[0]

            if act_name == "set_1_point":
                _, start_area, start_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([start_x, start_y])
                break
            elif act_name == "set_2_points":
                _, start_area, start_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_3_points":
                _, start_area, start_subarea, mid_area, mid_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_x, mid_y = area_to_xy(mid_area, mid_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_x, mid_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_4_points":
                _, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_1_x, mid_1_y = area_to_xy(mid_1_area, mid_1_subarea, rows, cols, height, width)
                mid_2_x, mid_2_y = area_to_xy(mid_2_area, mid_2_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_1_x, mid_1_y])
                trajectory.append([mid_2_x, mid_2_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_5_points":
                _, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, mid_3_area, mid_3_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_1_x, mid_1_y = area_to_xy(mid_1_area, mid_1_subarea, rows, cols, height, width)
                mid_2_x, mid_2_y = area_to_xy(mid_2_area, mid_2_subarea, rows, cols, height, width)
                mid_3_x, mid_3_y = area_to_xy(mid_3_area, mid_3_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_1_x, mid_1_y])
                trajectory.append([mid_2_x, mid_2_y])
                trajectory.append([mid_3_x, mid_3_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_6_points":
                _, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, mid_3_area, mid_3_subarea, mid_4_area, mid_4_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_1_x, mid_1_y = area_to_xy(mid_1_area, mid_1_subarea, rows, cols, height, width)
                mid_2_x, mid_2_y = area_to_xy(mid_2_area, mid_2_subarea, rows, cols, height, width)
                mid_3_x, mid_3_y = area_to_xy(mid_3_area, mid_3_subarea, rows, cols, height, width)
                mid_4_x, mid_4_y = area_to_xy(mid_4_area, mid_4_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_1_x, mid_1_y])
                trajectory.append([mid_2_x, mid_2_y])
                trajectory.append([mid_3_x, mid_3_y])
                trajectory.append([mid_4_x, mid_4_y])
                trajectory.append([end_x, end_y])
                break
            else:
                trajectory = []
                round_count -= 1
                time.sleep(5)
                continue
    
    trajectory = np.array(trajectory).astype(np.float32)

    output_image_trajectory = os.path.join(save_path, f"trajectory.png")
    plot_trajectory_multi([trajectory.astype(np.int64).tolist()], grid_image, output_image_trajectory)
    
    trajectory[..., 0] = np.clip(trajectory[..., 0], a_min=0, a_max=width-1) / (width-1)
    trajectory[..., 1] = np.clip(trajectory[..., 1], a_min=0, a_max=height-1) / (height-1)
    
    return trajectory.tolist()

def agent_single_trajectory_grounding(image_path, segment_path, save_path, task_desc, center_point, sample_point, mllm, rethink=False, video_path='', start_point_act_last='', trajectory_act_last='', round_max=10):
    start_area, start_subarea = "None", "None"
    output_image_point = ""

    grid_image = os.path.join(save_path, "grid.png")
    rows, cols, height, width = draw_grid(image_path, grid_image)

    round_count = 0
    while round_count < round_max:
        round_count += 1
        
        if not rethink:
            prompt = prompts.task_template_set_start_point_ground
            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            status, rsp = mllm.get_model_response(prompt, segment_path)
        else:
            prompt_rethink = prompts.task_template_rethinking
            prompt_rethink = re.sub(r"<text_prompt>", task_desc.replace(".",""), prompt_rethink)
            prompt = prompts.task_template_set_start_point_ground
            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            status, rsp = mllm.get_model_response_rethink(prompt, segment_path, prompt_rethink, video_path, start_point_act_last)

        if status:
            res, start_point_act = parse_grid_rsp_select(rsp)
        
            if res[0] == "ERROR":
                round_count -= 1
                time.sleep(5)
                continue
            
            act_name = res[0]

            if act_name == "select_object":
                _, object_idx = res
                if object_idx >= len(center_point):
                    round_count -= 1
                    time.sleep(5)
                    continue
                x, y = center_point[object_idx]
                sample_point = np.array(sample_point[object_idx])
                start_area, start_subarea = xy_to_area(x, y, rows, cols, height, width)
                x, y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                if not rethink:
                    output_image_point = os.path.join(save_path, f"start.png")
                else:
                    output_image_point = os.path.join(save_path, f"start_rethink.png")
                plot_trajectory([[x, y]], grid_image, output_image_point)
                break
            else:
                round_count -= 1
                time.sleep(5)
                continue
        else:
            round_count -= 1
            time.sleep(5)
            continue

    trajectory = []
    round_count = 0
    while round_count < round_max:
        round_count += 1

        if not rethink:
            prompt = prompts.task_template_trajectory_given_start
            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            start_point_location = f"({str(start_area)}, {start_subarea})"
            prompt = re.sub(r"<start_point_location>", start_point_location, prompt)
            status, rsp = mllm.get_model_response(prompt, output_image_point)
        else:
            prompt_rethink = prompts.task_template_rethinking
            
            prompt = prompts.task_template_trajectory_given_start
            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            start_point_location = f"({str(start_area)}, {start_subarea})"
            prompt = re.sub(r"<start_point_location>", start_point_location, prompt)
            status, rsp = mllm.get_model_response_rethink(prompt, output_image_point, prompt_rethink, video_path, trajectory_act_last)
            
        if status:
            res, trajectory_act = parse_grid_rsp_given_start(rsp)
        
            if res[0] == "ERROR":
                round_count -= 1
                time.sleep(5)
                continue
            
            act_name = res[0]

            if act_name == "set_1_point":
                _, start_area, start_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([start_x, start_y])
                break
            elif act_name == "set_2_points":
                _, start_area, start_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_3_points":
                _, start_area, start_subarea, mid_area, mid_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_x, mid_y = area_to_xy(mid_area, mid_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_x, mid_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_4_points":
                _, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_1_x, mid_1_y = area_to_xy(mid_1_area, mid_1_subarea, rows, cols, height, width)
                mid_2_x, mid_2_y = area_to_xy(mid_2_area, mid_2_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_1_x, mid_1_y])
                trajectory.append([mid_2_x, mid_2_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_5_points":
                _, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, mid_3_area, mid_3_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_1_x, mid_1_y = area_to_xy(mid_1_area, mid_1_subarea, rows, cols, height, width)
                mid_2_x, mid_2_y = area_to_xy(mid_2_area, mid_2_subarea, rows, cols, height, width)
                mid_3_x, mid_3_y = area_to_xy(mid_3_area, mid_3_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_1_x, mid_1_y])
                trajectory.append([mid_2_x, mid_2_y])
                trajectory.append([mid_3_x, mid_3_y])
                trajectory.append([end_x, end_y])
                break
            elif act_name == "set_6_points":
                _, start_area, start_subarea, mid_1_area, mid_1_subarea, mid_2_area, mid_2_subarea, mid_3_area, mid_3_subarea, mid_4_area, mid_4_subarea, end_area, end_subarea = res
                start_x, start_y = area_to_xy(start_area, start_subarea, rows, cols, height, width)
                mid_1_x, mid_1_y = area_to_xy(mid_1_area, mid_1_subarea, rows, cols, height, width)
                mid_2_x, mid_2_y = area_to_xy(mid_2_area, mid_2_subarea, rows, cols, height, width)
                mid_3_x, mid_3_y = area_to_xy(mid_3_area, mid_3_subarea, rows, cols, height, width)
                mid_4_x, mid_4_y = area_to_xy(mid_4_area, mid_4_subarea, rows, cols, height, width)
                end_x, end_y = area_to_xy(end_area, end_subarea, rows, cols, height, width)
                trajectory.append([start_x, start_y])
                trajectory.append([mid_1_x, mid_1_y])
                trajectory.append([mid_2_x, mid_2_y])
                trajectory.append([mid_3_x, mid_3_y])
                trajectory.append([mid_4_x, mid_4_y])
                trajectory.append([end_x, end_y])
                break
            else:
                trajectory = []
                round_count -= 1
                time.sleep(5)
                continue
        else:
            trajectory = []
            round_count -= 1
            time.sleep(5)
            continue
    
    trajectory = np.array(trajectory).astype(np.float32)
    trajectory -= trajectory[0:1]
    trajectory = sample_point[:, None] + trajectory[None]
    if not rethink:
        output_image_trajectory = os.path.join(save_path, f"trajectory.png")
    else:
        output_image_trajectory = os.path.join(save_path, f"trajectory_rethink.png")
    plot_trajectory_multi(trajectory.astype(np.int64).tolist()[0:1], image_path, output_image_trajectory)
    
    trajectory[..., 0] = np.clip(trajectory[..., 0], a_min=0, a_max=width-1) / (width-1)
    trajectory[..., 1] = np.clip(trajectory[..., 1], a_min=0, a_max=height-1) / (height-1)
    
    return trajectory.tolist(), start_point_act, trajectory_act