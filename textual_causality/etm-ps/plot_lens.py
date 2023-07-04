import json
import matplotlib.pyplot as plt
import numpy as np

with open("save/document_lengths.json", "r") as fp:
    doc_lens = json.load(fp)

doc_lens = np.array(list(doc_lens.values()))
print(doc_lens)
plt.hist(doc_lens, bins = "auto")
plt.title("Histogram of document lengths")
plt.savefig("save/doc_lens_hist.jpg")

print("Summary Statistics:")
print(f"Min = {np.mean(doc_lens)}, Max = {np.max(doc_lens)}, Median = {np.median(doc_lens)}")
print(f"Standard Deviation = {np.std(doc_lens)}")
print(f"1st Quartile = {np.quantile(doc_lens, 0.25)}, 3rd Quartile = {np.quantile(doc_lens, 0.75)}")
print(f"{np.mean(doc_lens > 800)} are longer than 800.")
