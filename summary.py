import json
raw_data_path = "./testrecord.json"
raw_instruction_path = "./instruction.json"
datas = json.load(open(raw_data_path,"r"))
instruction = json.load(open(raw_instruction_path,"r"))
summary = {
    'hammer_beat':{
        "cnt":0,
        "images":[],
        "origin_model":{"SSIM":0,"PSNR (dB)":0},
        "trained_model":{"SSIM":0,"PSNR (dB)":0},
        "large_model":{"SSIM":0,"PSNR (dB)":0},
    },
    "handover":{
        "cnt":0,
        "images":[],
        "origin_model":{"SSIM":0,"PSNR (dB)":0},
        "trained_model":{"SSIM":0,"PSNR (dB)":0},
        "large_model":{"SSIM":0,"PSNR (dB)":0},
    },
    "stack":{
        "cnt":0,
        "images":[],
        "origin_model":{"SSIM":0,"PSNR (dB)":0},
        "trained_model":{"SSIM":0,"PSNR (dB)":0},
        "large_model":{"SSIM":0,"PSNR (dB)":0},
    }
}

for data in datas:
    if data["prompt"].strip() in instruction["hammer_beat"]:
        summary["hammer_beat"]["origin_model"]["SSIM"] += data["origin_model"]["SSIM"]
        summary["hammer_beat"]["origin_model"]["PSNR (dB)"] += data["origin_model"]["PSNR (dB)"]
        summary["hammer_beat"]["trained_model"]["SSIM"] += data["trained_model"]["SSIM"]
        summary["hammer_beat"]["trained_model"]["PSNR (dB)"] += data["trained_model"]["PSNR (dB)"]
        summary["hammer_beat"]["large_model"]["SSIM"] += data["large_model"]["SSIM"]
        summary["hammer_beat"]["large_model"]["PSNR (dB)"] += data["large_model"]["PSNR (dB)"]
        summary["hammer_beat"]["cnt"] += 1
        summary["hammer_beat"]["images"].append(data["idx"])
    elif data["prompt"].strip() in instruction["handover"]:
        summary["handover"]["origin_model"]["SSIM"] += data["origin_model"]["SSIM"]
        summary["handover"]["origin_model"]["PSNR (dB)"] += data["origin_model"]["PSNR (dB)"]
        summary["handover"]["trained_model"]["SSIM"] += data["trained_model"]["SSIM"]
        summary["handover"]["trained_model"]["PSNR (dB)"] += data["trained_model"]["PSNR (dB)"]
        summary["handover"]["large_model"]["SSIM"] += data["large_model"]["SSIM"]
        summary["handover"]["large_model"]["PSNR (dB)"] += data["large_model"]["PSNR (dB)"]
        summary["handover"]["cnt"] += 1
        summary["handover"]["images"].append(data["idx"])
    elif data["prompt"].strip() in instruction["stack"]:
        summary["stack"]["origin_model"]["SSIM"] += data["origin_model"]["SSIM"]
        summary["stack"]["origin_model"]["PSNR (dB)"] += data["origin_model"]["PSNR (dB)"]
        summary["stack"]["trained_model"]["SSIM"] += data["trained_model"]["SSIM"]
        summary["stack"]["trained_model"]["PSNR (dB)"] += data["trained_model"]["PSNR (dB)"]
        summary["stack"]["large_model"]["SSIM"] += data["large_model"]["SSIM"]
        summary["stack"]["large_model"]["PSNR (dB)"] += data["large_model"]["PSNR (dB)"]
        summary["stack"]["cnt"] += 1
        summary["stack"]["images"].append(data["idx"])
    else:
        msg = f"Wrong instruction: {data["prompt"].strip()}"
        raise ValueError(msg)
    
summary["hammer_beat"]["origin_model"]["SSIM"] /= summary["hammer_beat"]["cnt"]
summary["hammer_beat"]["origin_model"]["PSNR (dB)"] /= summary["hammer_beat"]["cnt"]
summary["hammer_beat"]["trained_model"]["SSIM"] /= summary["hammer_beat"]["cnt"]
summary["hammer_beat"]["trained_model"]["PSNR (dB)"] /= summary["hammer_beat"]["cnt"]
summary["hammer_beat"]["large_model"]["SSIM"] /= summary["hammer_beat"]["cnt"]
summary["hammer_beat"]["large_model"]["PSNR (dB)"] /= summary["hammer_beat"]["cnt"]

summary["handover"]["origin_model"]["SSIM"] /= summary["handover"]["cnt"]
summary["handover"]["origin_model"]["PSNR (dB)"] /= summary["handover"]["cnt"]
summary["handover"]["trained_model"]["SSIM"] /= summary["handover"]["cnt"]
summary["handover"]["trained_model"]["PSNR (dB)"] /= summary["handover"]["cnt"]
summary["handover"]["large_model"]["SSIM"] /= summary["handover"]["cnt"]
summary["handover"]["large_model"]["PSNR (dB)"] /= summary["handover"]["cnt"]

summary["stack"]["origin_model"]["SSIM"] /= summary["stack"]["cnt"]
summary["stack"]["origin_model"]["PSNR (dB)"] /= summary["stack"]["cnt"]
summary["stack"]["trained_model"]["SSIM"] /= summary["stack"]["cnt"]
summary["stack"]["trained_model"]["PSNR (dB)"] /= summary["stack"]["cnt"]
summary["stack"]["large_model"]["SSIM"] /= summary["stack"]["cnt"]
summary["stack"]["large_model"]["PSNR (dB)"] /= summary["stack"]["cnt"]
with open("split_summary.json","w+") as f:
    json.dump(summary,f)

print(
f"""-----------------------------
hammer_beat:
    origin_model: 
    SSIM:     {summary["hammer_beat"]["origin_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["hammer_beat"]["origin_model"]["PSNR (dB)"]:.4f}
    trained_model: 
    SSIM:     {summary["hammer_beat"]["trained_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["hammer_beat"]["trained_model"]["PSNR (dB)"]:.4f}
    large_model: 
    SSIM:     {summary["hammer_beat"]["large_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["hammer_beat"]["large_model"]["PSNR (dB)"]:.4f}
-----------------------------
handover:
    origin_model: 
    SSIM:     {summary["handover"]["origin_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["handover"]["origin_model"]["PSNR (dB)"]:.4f}
    trained_model: 
    SSIM:     {summary["handover"]["trained_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["handover"]["trained_model"]["PSNR (dB)"]:.4f}
    large_model: 
    SSIM:     {summary["handover"]["large_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["handover"]["large_model"]["PSNR (dB)"]:.4f}
-----------------------------
stack:
    origin_model: 
    SSIM:     {summary["stack"]["origin_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["stack"]["origin_model"]["PSNR (dB)"]:.4f}
    trained_model: 
    SSIM:     {summary["stack"]["trained_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["stack"]["trained_model"]["PSNR (dB)"]:.4f}
    large_model: 
    SSIM:     {summary["stack"]["large_model"]["SSIM"]:.4f}
    PSNR (dB): {summary["stack"]["large_model"]["PSNR (dB)"]:.4f}
-----------------------------
"""
)