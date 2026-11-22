"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    # TODO Mission 3: a doorbell will send you glitches like the ones you found in Mission 1.
    # Give every field a sensible range, e.g. Field(ge=..., le=...), so a glitch is refused
    # with HTTP 422 instead of being answered with a made-up number. Base the limits on what a
    # real sensor can report, not on your data's min and max: real readings must still get answers.
    # Ranges follow what a real sensor/device can report (Li-ion cell, outdoor temps, Wi-Fi dBm),
    # wider than the data's min/max but excluding the glitch values (0 V, -127 C, 0 dBm).
    battery_v: float = Field(ge=2.5, le=4.4)
    temp_c: float = Field(ge=-40, le=85)
    rings_per_day: float = Field(ge=0, le=200)
    wifi_rssi_dbm: float = Field(ge=-100, le=-20)
    age_months: float = Field(ge=0, le=120)

    # A broken sensor can also send NaN or Infinity. Turn those into None so they fail
    # validation with a clean HTTP 422 (NaN itself can't be written back in a JSON error).
    from pydantic import field_validator

    @field_validator("*", mode="before")
    @classmethod
    def reject_nan_inf(cls, v):
        return None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    # TODO Mission 2 + 4: return your per-layer table, worked out FROM THE NETWORK (net.net),
    # not typed in by hand, so it stays right if you change the layer sizes. Exactly these keys:
    # {"layers": [{"name": "linear1", "in": 5, "out": 32, "params": ..., "bytes_f32": ..., "macs": ...}, ...],
    #  "total_params": ..., "total_bytes_f32": ..., "total_bytes_int8": ..., "total_macs": ...,
    #  "fits_300kb_sram_f32": true/false, "fits_300kb_sram_int8": true/false}
    # Count only the Linear layers (ReLU has no parameters). Clue: a Linear layer m has
    # m.in_features and m.out_features.
    layers = []
    for m in net.net:
        if isinstance(m, torch.nn.Linear):
            params = m.in_features * m.out_features + m.out_features   # weights + biases
            layers.append({
                "name": f"linear{len(layers) + 1}",
                "in": m.in_features,
                "out": m.out_features,
                "params": params,
                "bytes_f32": params * 4,                                # float32 = 4 bytes
                "macs": m.in_features * m.out_features,                 # one MAC per weight
            })

    total_params = sum(layer["params"] for layer in layers)
    sram_limit = 300 * 1024                                             # ~300 KB free SRAM

    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_params * 4,
        "total_bytes_int8": total_params * 1,                           # int8 = 1 byte
        "total_macs": sum(layer["macs"] for layer in layers),
        "fits_300kb_sram_f32": total_params * 4 <= sram_limit,
        "fits_300kb_sram_int8": total_params * 1 <= sram_limit,
    }


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json())})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]
    # TODO Mission 3: time each of the three stages below the same way "parse" is timed above,
    # storing them as t["normalise"], t["inference"] and t["build"].
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)
    t0 = time.perf_counter()
    with torch.no_grad():
        tensor_x = torch.from_numpy(x)
        if isinstance(data, Batch):
            y = np.array([
                net(tensor_x[i:i+1]).item()
                for i in range(len(rows))
            ], dtype=np.float32)
        else:
            y = net(tensor_x).numpy()
    t["inference"] = ms(t0)
    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)
