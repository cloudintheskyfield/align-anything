import os
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

# Set GPU 4 for inference
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_path = '/mnt/data3/nlp/ws/proj/align-anything/output/qwen_omni_sft/slice_end'

# Load model
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# Load processor correctly
processor = Qwen2_5OmniProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

print('Model and processor loaded successfully!')

# Example inference function
def inference(text_input, image_path=None):
    """
    Perform inference with the loaded model
    """
    if image_path:
        from PIL import Image
        image = Image.open(image_path)
        inputs = processor(text=text_input, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text_input, return_tensors="pt")
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=1,
            top_p=0.8,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

# Test inference
if __name__ == "__main__":
    # Simple text inference
    result = inference("Hello, how are you?")
    print(f"Response: {result}")

