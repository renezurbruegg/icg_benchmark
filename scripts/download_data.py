import shutil
import os
import gdown

resources = {
    "dataset": {
        "id": "1w5TkzRt-aQwLud1CKozbw6TBuwExVf50",
    },
    "icg_net": {
        "id": "1RzdhOBX5GV42qbvmjuL31Nf1N90gT-ra",
    },
    "edge_grasp_net": {
        "id": "1hmL5zeByQuWlytW1ZDDLyHy4pLjN-3VN",
    },
}

os.makedirs("data", exist_ok=True)

for key, values in resources.items():
    print("Downloading data for", key)
    output_name = "tmp.zip"
    gdown.download(id=values["id"], output=output_name, quiet=False)
    shutil.unpack_archive(output_name, "data")
    # remove the zip file
    os.remove(output_name)

print("Done downloading data.")
