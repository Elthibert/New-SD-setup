import os
import requests
import json

# Define the base path for Stable Diffusion
sd_path = '/workspace/stable-diffusion-webui'
models_path = os.path.join(sd_path, 'models')
adetailer_path = os.path.join(models_path, 'adetailer')
esrgan_path = os.path.join(models_path, 'ESRGAN')
lora_path = os.path.join(models_path, 'Lora')
stable_diffusion_path = os.path.join(models_path, 'Stable-diffusion')

# Ensure all directories exist
for path in [models_path, adetailer_path, esrgan_path, lora_path, stable_diffusion_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Function to download files
def download_file(url, path):
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)

# Download files for adetailer
download_file('https://huggingface.co/guon/hand-eyes/resolve/main/full_eyes_detect_v1.pt', os.path.join(adetailer_path, 'full_eyes_detect_v1.pt'))
download_file('https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt', os.path.join(adetailer_path, 'face_yolov8m.pt'))

# Download files for ESRGAN
download_file('https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/1x-ITF-SkinDiffDetail-Lite-v1.pth', os.path.join(esrgan_path, '1x-ITF-SkinDiffDetail-Lite-v1.pth'))

# Download files for Lora
download_file('https://civitai.com/api/download/models/534756', os.path.join(lora_path, 'PonyAmateur.safetensors'))
download_file('https://civitai.com/api/download/models/382152', os.path.join(lora_path, 'ExpressiveH.safetensors'))
download_file('https://civitai.com/api/download/models/383563', os.path.join(lora_path, 'ExtremelyDetailedSlider.safetensors'))
download_file('https://civitai.com/api/download/models/449162', os.path.join(lora_path, 'SinfullyStylishDramaLight.safetensors'))
download_file('https://civitai.com/api/download/models/449028', os.path.join(lora_path, 'ThePitStyle.safetensors'))

# Download files for Stable-diffusion
download_file('https://civitai.com/api/download/models/534642', os.path.join(stable_diffusion_path, 'PonyRealismV2_1.safetensors'))
download_file('https://civitai.com/api/download/models/344487', os.path.join(stable_diffusion_path, 'RealVisXL.safetensors'))

# Define your preferred settings
custom_settings = {
    "sd_model_checkpoint": "PonyRealismV2_1.safetensors",
    "CLIP_stop_at_last_layers": 2,
    "sampler_name": "DPM++ SDE Karras",
    "steps": 35,
    "width": 768,
    "height": 1280,
    "cfg_scale": 7,
    "batch_size": 1,
    "batch_count": 1,
    "seed": -1,
    "refiner_checkpoint": "RealVisXL.safetensors",
    "refiner_switch_at": 0.8,
    "enable_hr": True,
    "hr_upscaler": "1x-ITF-SkinDiffDetail-Lite-v1",
    "hr_second_pass_steps": 35,
    "denoising_strength": 0.4,
    "hr_scale": 2,
    "ad_model": "face_yolov8m.pt,full_eyes_detect_v1.pt",
}

# Create a custom config file
config_path = os.path.join(sd_path, 'config.json')
with open(config_path, 'w') as f:
    json.dump(custom_settings, f, indent=4)
print(f"Custom config saved to {config_path}")

# Create prompt templates
prompts = {
    "positive": "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, BREAK,(1girl, 18 years old, freckles),(horny face, horny eyes, horny looking:1.4), blue eyes, intense gaze,(brunette,long hair, eyeliner, blush, lipstick:1.4), amateur, raw, instagram photo, amateur photo, traditional media <lora:AmateurStyle_v1_PONY_REALISM:.3>",
    "negative": "score_1, score_2, score_3, (tattoo:1.5), ink, deformed, deformed face, low quality, bad quality, worst quality, (drawn, furry, illustration, cartoon, anime, comic:1.5), 3d, cgi, extra fingers, (source_anime, source_cartoon, source_furyy, source_western, source_comic, source_pony)",
    "ad_positive1": "(1girl, 18 years old, freckles),(horny face, horny eyes, horny looking:1.4), blue eyes, intense gaze,(brunette,long hair, eyeliner, blush, lipstick:1.4), amateur, raw, instagram photo",
    "ad_negative1": "score_1, score_2, score_3, (tattoo:1.5), ink, deformed, deformed face, low quality, bad quality, worst quality, (drawn, furry, illustration, cartoon, anime, comic:1.5), 3d, cgi, extra fingers, (source_anime, source_cartoon, source_furyy, source_western, source_comic, source_pony)",
    "ad_positive2": "(1girl, 18 years old),blue eyes, raw",
    "ad_negative2": "score_1, score_2, score_3, (tattoo:1.5), ink, deformed, deformed face, low quality, bad quality, worst quality, (drawn, furry, illustration, cartoon, anime, comic:1.5), 3d, cgi, extra fingers, (source_anime, source_cartoon, source_furyy, source_western, source_comic, source_pony)",
}

prompts_path = os.path.join(sd_path, 'prompt_templates.json')
with open(prompts_path, 'w') as f:
    json.dump(prompts, f, indent=4)
print(f"Prompt templates saved to {prompts_path}")

print("Setup complete!")
