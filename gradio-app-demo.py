#!/usr/bin/env python
# coding: utf-8

import gradio as gr
from ultralytics import YOLO


def image_classifier(inp):
    label_map = {
        0: 'dead',
        1: 'live_animal',
        2: 'other',
        3: 'scat',
        4: 'tracks'
    }

    res = model(inp)
    probs = res[0].probs
    confidences = probs.top5conf.cpu().numpy().tolist()

    d = {label_map[k]: v for k, v in zip(probs.top5, confidences)}
    return d, d


def main():
    label = gr.Label(num_top_classes=5)
    input_image = gr.Image(sources=['upload'])
    examples = [
        "https://biodiv.app/ainaturalistype/examples/dead.png",
        "https://biodiv.app/ainaturalistype/examples/live_animal.png",
        "https://biodiv.app/ainaturalistype/examples/other.png",
        "https://biodiv.app/ainaturalistype/examples/scat.png",
        "https://biodiv.app/ainaturalistype/examples/tracks.png"
    ]

    demo = gr.Interface(fn=image_classifier,
                        inputs=input_image,
                        outputs=[label, 'json'],
                        allow_flagging='manual',
                        title='AiNaturalisType',
                        examples=examples)
    demo.launch()


if __name__ == '__main__':
    model = YOLO('AiNaType-v8x.pt')
    main()
