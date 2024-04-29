#!/usr/bin/env python
# coding: utf-8

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ----------------------------------------------------------------------------

app = FastAPI()

# ----------------------------------------------------------------------------


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        help='Path/URL to the weights file',
                        type=str)
    parser.add_argument('-m',
                        '--model-version',
                        help='Model version',
                        type=str)
    parser.add_argument('-H',
                        '--host',
                        help='API host (default: 0.0.0.0)',
                        type=str,
                        default='0.0.0.0')
    parser.add_argument('-s',
                        '--port',
                        help='API port (default: 8000)',
                        type=int,
                        default=8011)
    return parser.parse_args()


# ----------------------------------------------------------------------------


class Task(BaseModel):
    task: dict
    project: Optional[int] = None


def load_model(model_weights: str, model_version: str) -> dict:
    model_obj = YOLO(model_weights)
    model_dict = {'model': model_obj, 'model_version': model_version}
    return model_dict


def _pred_dict(model_version: str, score: float, label: str) -> dict:
    return {
        'type': 'choices',
        'score': score,
        'value': {
            'choices': [label]
        },
        'to_name': 'image',
        'from_name': 'choice',
        'model_version': model_version
    }


@app.post('/predict')
def predict_endpoint(task: Task):
    _task = task.task
    if not _task.get('project'):
        if task.project:
            if task.project not in MODEL.keys():
                raise HTTPException(
                    404, f'Project id `{task.project}` does not exist!')
            _task['project'] = task.project
        else:
            raise HTTPException(
                404, 'Parameter `project` is required when the task does not '
                'contain a project id number!')
    task = _task

    model_version = MODEL['model_version']
    model = MODEL['model']

    image_url = task['data']['image']
    img = Path(image_url)

    with tempfile.NamedTemporaryFile(suffix=img.suffix) as f:
        r = requests.get(image_url)
        if r.status_code == 200:
            f.write(r.content)
        else:
            return JSONResponse(content=r.text, status_code=404)
        f.seek(0)

        res = model(f.name)

    pred_label = res[0].names[res[0].probs.top1]
    pred_conf = res[0].probs.top1conf.cpu().numpy().tolist()

    if pred_conf < 0.70:
        pred_label = 'low_conf'

    result = [_pred_dict(model_version, pred_conf, pred_label)]
    pred = {'result': result}
    return JSONResponse(status_code=200, content=pred)


# ----------------------------------------------------------------------------

if __name__ == '__main__':
    args = opts()

    MODEL = load_model(args.weights, args.model_version)

    uvicorn.run(app, host=args.host, port=args.port)  # noqa
