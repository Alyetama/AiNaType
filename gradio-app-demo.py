#!/usr/bin/env python
# coding: utf-8

import gradio as gr
from ultralytics import YOLO


def image_classifier(inp):
    res = model(inp)
    p = res[0].probs
    d = {}
    for k, v in zip(p.top5, p.top5conf.cpu().numpy()):
        if k == 0:
            d['dead'] = v
        elif k == 1:
            d['live_animal'] = v
        elif k == 2:
            d['scat'] = v
        elif k == 3:
            d['tracks'] = v
    return d, d


def main():
    label = gr.Label(num_top_classes=4)
    input_image = gr.Image(sources=['upload'])

    demo = gr.Interface(fn=image_classifier,
                        inputs=input_image,
                        outputs=[label, 'json'],
                        allow_flagging='manual')
    demo.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    model = YOLO('../yolov8/AiNaturalisType/ainatype-c4edQ/weights/best.pt')
    main()
