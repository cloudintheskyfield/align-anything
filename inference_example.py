#!/usr/bin/env python3
"""
Example inference script for Qwen2.5-Omni model
Usage:
    python inference_example.py --text "Hello, how are you?"
    python inference_example.py --text "Describe this image" --image path/to/image.jpg
"""

import argparse
import torch
from PIL import Image
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

def load_model(model_path):
    """Load model and processor"""
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("Model and processor loaded successfully!")
    return model, processor

def inference(model, processor, text_input, image_path=None):
    """Perform inference"""
    print(f"Input text: {text_input}")
    
    if image_path:
        print(f"Input image: {image_path}")
        image = Image.open(image_path)
        inputs = processor(text=text_input, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text_input, return_tensors="pt")
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Remove input text from response
    if text_input in response:
        response = response.replace(text_input, "").strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Inference")
    parser.add_argument("--model_path", default="/mnt/data3/nlp/ws/proj/align-anything/output/qwen_omni_sft/slice_end", 
                       help="Path to the model")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--image", help="Path to input image (optional)")
    parser.add_argument("--gpu", default="4", help="GPU device to use")
    
    args = parser.parse_args()
    
    # Set GPU device
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load model
    model, processor = load_model(args.model_path)
    
    # Perform inference
    result = inference(model, processor, args.text, args.image)
    
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(result)
    print("="*50)

if __name__ == "__main__":
    main()
