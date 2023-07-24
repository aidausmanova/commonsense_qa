import os
import time
import torch
import carbontracker
from transformers import T5ForConditionalGeneration, T5Tokenizer
from carbontracker.tracker import CarbonTracker
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# Define the path to the test set
root_path = "/export/home/0usmanov/project/output/"
# model_name = "t5s_nopretrain_Mar14_09-45-30" # t5s FT
# model_name = "t5s_adamw_mlm_Mar12_13-48-57_Mar13_23-02-45" # t5s FT+PT
model_name = "t5b_nopretrain_Mar14_13-46-51" # t5b FT
# model_name = "t5b_adamw_mlm_Mar16_11-31-25_Mar17_10-59-22" # t5b FT+PT
model_path = root_path + 'tellmewhy/checkpoints/' + model_name

tellmewhy = load_dataset('StonyBrookNLP/tellmewhy')
test_data = tellmewhy['test']

model = T5ForConditionalGeneration.from_pretrained(model_path + '/pytorch_model.bin',local_files_only=True, config=model_path + '/config.json')
tokenizer = T5Tokenizer.from_pretrained(model_path)

seeds = [42, 123, 456]
num_runs = len(seeds)

efficiency_res = []

for i in range(num_runs):
    seed = seeds[i]
    torch.manual_seed(seed)
    start_time = time.time()
    print(f"Start run {i} for seed {seed} at ", time.strftime("%b%d_%H-%M-%S", time.localtime(start_time)))
    tracker = CarbonTracker(epochs=len(test_data), log_dir=root_path+"carbontracker/inference/")
    
    for j, input_row in enumerate(tqdm(test_data)):
        tracker.epoch_start()

        input_ids = tokenizer(
            input_row["question"],
            input_row["narrative"],
            max_length=396,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        output_ids = model.generate(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            num_beams=1,
            max_length=80,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True
        )
        
        output_text = [
            tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in output_ids
        ]
        
        # Query nvidia-smi
        power_draw = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits").read().strip()
        # power_draw = os.popen("nvidia-smi --query-gpu=index,gpu_bus_id,power.draw --format=csv").read().strip()
        efficiency_res.append({
            "inference_run": i+1,
            "seed": seed,
            "input_id": j+1,
            "power_draw": power_draw
        })
        
        tracker.epoch_end()
        
        # Wait before running next inference
        time.sleep(5)
    
    # end_time = time.time()
    
    # elapsed_time = end_time - start_time
    # print("Elapsed time: ", time.strftime("%b%d_%H-%M-%S", time.localtime(elapsed_time)))
    # power_consumption_start = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits").read().strip()
    time.sleep(5)
    # power_consumption_end = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits").read().strip()
    # power_consumption = (float(power_consumption_start) + float(power_consumption_end)) / 2.0
    # print(f"Run {i} power consumption ", power_consumption)

output_df = pd.DataFrame(efficiency_res)
output_df.to_csv(f"{root_path}inference/{model_name}.csv")
print(f"Finished run for seed {seed} at ", time.strftime("%b%d_%H-%M-%S", time.localtime(time.time())))
