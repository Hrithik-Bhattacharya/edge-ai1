"""check_endpoint.py -- self-test mimicking the README checks.  python check_endpoint.py --url http://localhost:8000"""
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--url",default="http://localhost:8000"); ap.add_argument("--team",type=int,default=22); A=ap.parse_args()
import requests, numpy as np, pandas as pd, json
u=A.url.rstrip("/"); ok=lambda c,m: print("PASS" if c else "FAIL", m)
h=requests.get(u+"/health").json(); ok(h["status"]=="ok" and h["team"]==A.team,f"health {h}")
mi=requests.get(u+"/model_info").json(); L=mi["layers"]
ok(all(l["params"]==l["in"]*l["out"]+l["out"] and l["bytes_f32"]==4*l["params"] and l["macs"]==l["in"]*l["out"] for l in L),"layer math")
ok(L[0]["in"]==5 and L[-1]["out"]==1 and all(L[i]["out"]==L[i+1]["in"] for i in range(len(L)-1)),"layers connect")
ok(mi["total_params"]==h["params"]==sum(l["params"] for l in L) and mi["total_bytes_int8"]==mi["total_params"] and mi["total_macs"]==sum(l["macs"] for l in L),"totals")
ok(mi["fits_300kb_sram_f32"] is True and mi["fits_300kb_sram_int8"] is True,"fits flags")
g=dict(battery_v=3.8,temp_c=28,rings_per_day=5,wifi_rssi_dbm=-66,age_months=22)
bad=[{**g,"battery_v":0},{**g,"temp_c":-127},{**g,"wifi_rssi_dbm":0},{**g,"rings_per_day":-3},{**g,"age_months":-5},
     {**g,"battery_v":9},{**g,"temp_c":300},{k:v for k,v in g.items() if k!="temp_c"},{**g,"battery_v":"abc"},{**g,"battery_v":None}]
print([requests.post(u+"/predict",json=b).status_code for b in bad])
raw=requests.post(u+"/predict",data='{"battery_v":NaN,"temp_c":28,"rings_per_day":5,"wifi_rssi_dbm":-66,"age_months":22}'); print("NaN",raw.status_code)
near=[{**g,"battery_v":4.25},{**g,"temp_c":50},{**g,"rings_per_day":60},{**g,"wifi_rssi_dbm":-95},{**g,"age_months":40}]
print("just-outside real readings:",[requests.post(u+"/predict",json=b).status_code for b in near])
df=pd.read_csv("data/team_%02d_train.csv"%A.team+"").iloc[:50]; rows=df[["battery_v","temp_c","rings_per_day","wifi_rssi_dbm","age_months"]].to_dict("records")
rows=[r for r in rows if r["battery_v"]>0 and r["temp_c"]>-75 and r["wifi_rssi_dbm"]<0]
s=[requests.post(u+"/predict",json=r).json() for r in rows]
b=requests.post(u+"/predict_batch",json={"rows":rows}).json()
ok(b["days_left"]==[x["days_left"] for x in s],f"batch == {len(rows)} singles")
need={"parse","normalise","inference","build","server_total"}
ok(all(need<=set(x["timings_ms"]) for x in s) and need<=set(b["timings_ms"]),"timings keys")
