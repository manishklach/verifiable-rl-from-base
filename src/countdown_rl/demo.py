"""Interactive base-versus-RL comparison with live verification."""

from __future__ import annotations

import argparse

import torch

from .evaluate import load_model
from .prompts import make_prompt
from .verifier import verify_completion


def main() -> None:
    import gradio as gr

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--trained", required=True)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    models = {"Base": load_model(args.base), "GRPO": load_model(args.trained)}

    def run(numbers: str, target: int):
        nums = [int(value.strip()) for value in numbers.split(",") if value.strip()]
        if not 3 <= len(nums) <= 5:
            raise gr.Error("Enter between three and five comma-separated integers")
        prompt = make_prompt(nums, int(target))
        outputs = []
        for label, (model, tokenizer) in models.items():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                sequence = model.generate(**inputs, max_new_tokens=256, do_sample=False)[0]
            completion = tokenizer.decode(
                sequence[inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            check = verify_completion(completion, nums, int(target))
            outputs.append(
                (label, completion, "✅ Correct" if check.correct else f"❌ {check.error}")
            )
        return outputs[0][1], outputs[0][2], outputs[1][1], outputs[1][2]

    with gr.Blocks(title="Verifiable RL from Base") as app:
        gr.Markdown(
            "# Verifiable RL from Base\nCompare the untouched base model with its GRPO adapter."
        )
        with gr.Row():
            numbers = gr.Textbox(value="8, 3, 4", label="Numbers")
            target = gr.Number(value=20, precision=0, label="Target")
            button = gr.Button("Solve", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Base")
                base_output = gr.Textbox(lines=10, label="Completion")
                base_verdict = gr.Textbox(label="Verifier")
            with gr.Column():
                gr.Markdown("## GRPO")
                trained_output = gr.Textbox(lines=10, label="Completion")
                trained_verdict = gr.Textbox(label="Verifier")
        button.click(
            run, [numbers, target], [base_output, base_verdict, trained_output, trained_verdict]
        )
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
