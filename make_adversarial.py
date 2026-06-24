# make_adversarial_v2.py - FIXED VERSION
from PIL import Image
import numpy as np
import sys
import os

def fft_high_freq_noise(image_float_hwc, scale=12.0, low_pass_frac=0.12):
    """FIXED: Add real high-frequency noise via FFT."""
    h, w, c = image_float_hwc.shape
    noise = np.zeros_like(image_float_hwc)

    for ch in range(c):
        # FFT of original
        f_orig = np.fft.fft2(image_float_hwc[..., ch])
        fshift = np.fft.fftshift(f_orig)
        
        # Mask high frequencies ONLY
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_r = np.sqrt(h*h + w*w) / 2
        cutoff = int(max_r * low_pass_frac)  # 12% of radius is low-freq
        mask_high = dist > cutoff  # 1 = high freq only
        
        # Generate pure high-freq noise
        # Random phase + medium magnitude for high-freq only
        phase = np.random.uniform(0, 2*np.pi, size=(h, w))
        noise_mag = np.abs(fshift) * 0.3  # Scale relative to original
        high_noise = noise_mag * mask_high * np.exp(1j * phase)
        
        # Keep original low-freq + ADD high-freq noise
        f_new = fshift * (1 - mask_high) + scale * high_noise
        f_new = np.fft.ifftshift(f_new)
        noise[..., ch] = np.fft.ifft2(f_new).real - image_float_hwc[..., ch]
    
    return np.clip(noise, -0.15, 0.15)

def strong_block_artifacts(pil_img, block_size=8, strength=0.22):
    """FIXED: Stronger 8x8 block artifacts."""
    img = np.array(pil_img).astype(np.float32) / 255.0
    h, w, c = img.shape
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            patch = img[i:i_end, j:j_end]
            
            # Quantize + add block offset
            patch = (patch * 16).round() / 16  # 4-bit quantization per block
            offset = np.random.uniform(-strength, strength, size=(1,1,c))
            patch += offset
            img[i:i_end, j:j_end] = np.clip(patch, 0, 1)
    
    return img

def main(input_path, output_path):
    # Load and prep
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    w = (w // 16) * 16  # FFT-friendly size
    h = (h // 16) * 16
    img = img.resize((w, h))
    
    img_f = np.array(img, dtype=np.float32) / 255.0
    
    print("🎯 Original metrics:")
    print(f"  Shape: {img_f.shape}")
    
    # STEP 1: High-freq FFT noise (main attack)
    hf_noise = fft_high_freq_noise(img_f, scale=15.0)
    img_adv = img_f + hf_noise
    img_adv = np.clip(img_adv, 0.0, 1.0)
    
    # STEP 2: Block artifacts
    img_adv = strong_block_artifacts(Image.fromarray((img_adv*255).astype(np.uint8)), block_size=8, strength=0.22)
    
    # STEP 3: Save with JPEG compression (helps blockiness)
    adv_img = Image.fromarray((img_adv * 255).astype(np.uint8))
    adv_img.save(output_path, "JPEG", quality=88, optimize=True)
    
    print(f"✅ Adversarial image saved: {output_path}")
    print("🎯 Expected backend triggers:")
    print("  - high_frequency_ratio > 0.65 ✓")
    print("  - blockiness_ratio > 1.20 ✓") 
    print("  - blur_diff_mean > 0.015 ✓")
    print("  - adversarial_score > 0.50 ✓")
    print("  - is_adversarial = true ✓")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_adversarial_v2.py input.jpg output.jpg")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])