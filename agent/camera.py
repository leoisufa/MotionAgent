import re
import time
import agent.prompts as prompts
from agent.agent_utils import parse_camera_rsp

def agent_camera_motion(image_path, task_desc, mllm, rethink=False, video_path='', camera_act_last='', round_max=1):
    round_count = 0

    while round_count < round_max:
        round_count += 1

        if not rethink:
            prompt = prompts.task_template_camera

            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            status, rsp = mllm.get_model_response(prompt, image_path)
        else:
            prompt_rethink = prompts.task_template_rethinking
            
            prompt = prompts.task_template_camera
            prompt = re.sub(r"<task_description>", task_desc.replace(".",""), prompt)
            status, rsp = mllm.get_model_response_rethink(prompt, image_path, prompt_rethink, video_path, camera_act_last)

        if status:
            res, camera_act = parse_camera_rsp(rsp)
        
            if res[0] == "ERROR":
                round_count -= 1
                time.sleep(5)
                continue
            
            act_name = res[0]

            camera_motion = []
            if act_name == "set_camera_motion":
                x_translation, y_translation, z_translation, x_rotation, y_rotation, z_rotation, motion_type = res[1:]
                if motion_type == "uniform":
                    camera_motion.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    camera_motion.append([x_translation, y_translation, z_translation, x_rotation, y_rotation, z_rotation])
                elif motion_type == "decrement":
                    camera_motion.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    camera_motion.append([x_translation*0.5, y_translation*0.5, z_translation*0.5, x_rotation*0.5, y_rotation*0.5, z_rotation*0.5])
                    camera_motion.append([x_translation*0.85, y_translation*0.85, z_translation*0.85, x_rotation*0.85, y_rotation*0.85, z_rotation*0.85])
                    camera_motion.append([x_translation, y_translation, z_translation, x_rotation, y_rotation, z_rotation])
                elif motion_type == "increment":
                    camera_motion.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    camera_motion.append([x_translation*0.15, y_translation*0.15, z_translation*0.15, x_rotation*0.15, y_rotation*0.15, z_rotation*0.15])
                    camera_motion.append([x_translation*0.5, y_translation*0.5, z_translation*0.5, x_rotation*0.5, y_rotation*0.5, z_rotation*0.5])
                    camera_motion.append([x_translation, y_translation, z_translation, x_rotation, y_rotation, z_rotation])
                else:
                    round_count -= 1
                    time.sleep(5)
                    continue
                return camera_motion, camera_act
            elif act_name == "set_camera_motion_complex":
                x_shift_1, x_shift_2, y_shift_1, y_shift_2, z_shift_1, z_shift_2, x_rotation, y_rotation, z_rotation = res[1:]
                camera_motion.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                camera_motion.append([x_shift_1, y_shift_1, z_shift_1, x_rotation//2, y_rotation//2, z_rotation//2])
                camera_motion.append([x_shift_1+x_shift_2, y_shift_1+y_shift_2, z_shift_1+z_shift_2, x_rotation, y_rotation, z_rotation])
                return camera_motion, camera_act
        else:
            round_count -= 1
            time.sleep(5)
            continue