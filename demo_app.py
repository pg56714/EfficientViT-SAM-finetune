import os
import torch
import numpy as np
import gradio as gr
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.samcore.data_provider.utils import ResizeLongestSide, Normalize_and_Pad

# Initialize SAM model
sam_model = create_sam_model(name="xl1", weight_url="./checkpoints/200_0176.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sam_model.to(device=device)
sam_model.eval()


def generate_mask_sam(frame):
    image = np.array(frame)[:, :, :3]

    H, W, _ = image.shape
    bboxes = np.array([[0, 0, W, H]])

    sam_trans = ResizeLongestSide(sam_model.image_size[0])
    original_image_size = image.shape[:2]
    image_tensor = torch.from_numpy(image).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    resize_image = sam_trans.apply_image(image_tensor, original_image_size).squeeze(0)
    input_image_torch = resize_image.contiguous()
    input_size = tuple(input_image_torch.shape[-2:])

    with torch.no_grad():
        normalize_and_pad = Normalize_and_Pad(sam_model.image_size[0])

        box = torch.from_numpy(bboxes).to(torch.float32)
        box = sam_trans.apply_boxes(box, original_image_size).to(device)

        box_torch = torch.as_tensor(box, dtype=torch.float, device=device).unsqueeze(0)

        processed_image = normalize_and_pad({
            "image": input_image_torch,
            "masks": torch.empty((0, *input_image_torch.shape[-2:]), dtype=torch.float32),
            "points": torch.tensor([]),
            "bboxs": box,
            "shape": input_image_torch.shape[-2:]
        })["image"]
        processed_image = processed_image.to(device)

        processed_image = processed_image.unsqueeze(0)
        
        image_embedding = sam_model.image_encoder(processed_image)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
    upscaled_masks = sam_model.postprocess_masks(
        low_res_masks, input_size, original_image_size
    ).to(device)
    thresholded_mask = (upscaled_masks > 0.5)[0].detach().squeeze(0).cpu().numpy()
    thresholded_mask = np.array(thresholded_mask * 255).astype(np.uint8)

    return thresholded_mask


title = """<p><h1 align="center">Test Demo</h1></p>"""
description = """<p>Test<p>"""


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            frame = gr.Image()
            mask_button = gr.Button("Submit")
        mask = gr.Image()
    mask_button.click(fn=generate_mask_sam, inputs=frame, outputs=mask)

demo.launch(debug=True, show_error=True)
