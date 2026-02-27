import gradio as gr


def build_ui(run_fn):
    with gr.Blocks() as demo:
        gr.Markdown("## 🎞️ Vintage Car Video Generator (CPU Only)")

        prompt = gr.Textbox(
            label="Prompt",
            placeholder="A red vintage car driving on a rainy road"
        )

        output = gr.Video(label="Generated Video")
        status = gr.Textbox(label="Status")

        generate_btn = gr.Button("Generate")

        generate_btn.click(
            run_fn,
            inputs=prompt,
            outputs=[output, status]
        )

    return demo
