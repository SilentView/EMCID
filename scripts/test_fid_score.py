import os
import json
import argparse
from pytorch_fid import fid_score

def main():
    parser = argparse.ArgumentParser(description="Calculate FID score between generated images and COCO-30K dataset.")
    parser.add_argument("--generated_images_folder", type=str, default="data/coco/images/sd_orig")
    parser.add_argument("--coco_30k_folder", type=str, default="data/coco/coco-30k")
    parser.add_argument("--output_folder", type=str, default="results/sd_orig")
    parser.add_argument("--output_file", type=str, default="fid.json")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for data loading.")
    parser.add_argument("--save_npz_folder", default="data/stats",help="Path to the folder to save the npz.")
    parser.add_argument("--device", required=False,default="cuda:0",help="cuda device")
    parser.add_argument("--dict_key", default="sd_orig", help="The key of the result dict.")
    args = parser.parse_args()
    
    generated_images_folder = args.generated_images_folder
    coco_30k_folder = args.coco_30k_folder
    output_folder = args.output_folder
    batch_size = args.batch_size
    save_npz_folder = args.save_npz_folder
    device = args.device
    last_folder_name = os.path.basename(os.path.normpath(generated_images_folder))
    #fid_score = calculate_fid_score(generated_images_folder, coco_30k_folder, batch_size=batch_size)

    if not os.path.exists(save_npz_folder):
        os.makedirs(save_npz_folder)
    
    if args.save_npz_folder:
        if not os.path.exists(save_npz_folder+f"/{last_folder_name}.npz"):
            fid_score.save_fid_stats([generated_images_folder,save_npz_folder+f"/{last_folder_name}.npz"], batch_size, device, dims=2048, num_workers=1)
        if not os.path.exists(save_npz_folder+f"/coco30k.npz"):
            fid_score.save_fid_stats([coco_30k_folder,save_npz_folder+f"/coco30k.npz"], batch_size, device, dims=2048, num_workers=1)
        fid_value = fid_score.calculate_fid_given_paths([save_npz_folder+f"/{last_folder_name}.npz", save_npz_folder+f"/coco30k.npz"],
                                                 batch_size,
                                                 device=device,
                                                 dims=2048)
    else:
        fid_value = fid_score.calculate_fid_given_paths([generated_images_folder, coco_30k_folder],
                                                    batch_size,
                                                    device=device,
                                                    dims=2048)
    result_dict = {"fid" : fid_value}
    print(result_dict)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the result as a JSON file
    output_file = os.path.join(output_folder, args.output_file)
    if os.path.exists(output_file):
        with open(output_file, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {}
    
    if args.dict_key in data:
        data[args.dict_key].update(result_dict)
    else:
        data[args.dict_key] = result_dict

    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    main()
