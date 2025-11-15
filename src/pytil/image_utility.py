from PIL import Image

def upscale_image(image_path, scale_factor, output_path):
    # Open the image
    with Image.open(image_path) as img:
        # Get the original dimensions
        original_width, original_height = img.size
        # Calculate new dimensions
        new_width = original_width * scale_factor
        new_height = original_height * scale_factor
        # Resize using NEAREST neighbor to preserve pixelation
        upscaled_img = img.resize((new_width, new_height), Image.NEAREST)
        # Save the upscaled image
        upscaled_img.save(output_path)