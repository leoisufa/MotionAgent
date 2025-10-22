from abc import abstractmethod
from typing import List
import requests
from decord import VideoReader, cpu
from PIL import Image
from agent.agent_utils import encode_image, encode_image_pillow


class BaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        pass


class OpenAIModel(BaseModel):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model_response(self, prompt: str, images: str) -> (bool, str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        content = [{"type": "text", "text": prompt}]
        if images:
            base64_img = encode_image(images)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}
            })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=300).json()
        except Exception as e:
            return False, str(e)

        if "error" not in response:
            usage = response["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            print(f"Request cost is "
                             f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03)}",
                             "yellow")
        else:
            return False, response["error"]["message"]

        return True, response["choices"][0]["message"]["content"]

    def get_model_response_rethink(self, prompt: str, images: str, prompt_rethink: str, video: str, action: str) -> (bool, str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        content = [{"type": "text", "text": prompt_rethink}]
        content.append({"type": "text", "text": "Following images are partial frames from the generated video:"})

        # 采样视频帧
        if video:
            vr = VideoReader(video, ctx=cpu(0))
            for i in range(0, len(vr), max(1, len(vr)//6)):
                frame = vr[i]
                frame_pillow = Image.fromarray(frame.asnumpy())
                frame_pillow.thumbnail((384, 384))
                base64_img = encode_image_pillow(frame_pillow)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}
                })

        # 追加上下文信息
        content.extend([
            {"type": "text", "text": f"The task you should recomplete is: {prompt}"},
            {"type": "text", "text": f"The action you made at last time is: {action}"}
        ])

        if images:
            base64_img = encode_image(images)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"}
            })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=300).json()
        except Exception as e:
            return False, str(e)

        if "error" not in response:
            usage = response["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            print(f"Request cost is "
                             f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03)}",
                             "yellow")
        else:
            return False, response["error"]["message"]

        return True, response["choices"][0]["message"]["content"]