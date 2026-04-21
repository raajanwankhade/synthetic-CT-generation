from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch


class Stage2VLM:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.2
        )

        result = self.processor.batch_decode(
            output, skip_special_tokens=True
        )[0]

        return result
