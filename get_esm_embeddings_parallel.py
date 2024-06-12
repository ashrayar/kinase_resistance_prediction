import torch
import esm
import sys

# Load ESM-2 model

base_dir="/wynton/home/fraserlab/aravikumar/dms/"
filename = sys.argv[1]
print("batch converting")
infile = open(base_dir+"met_var_sequences/"+filename)
data = []
for line in infile:
    lineparts=line[0:-1].split(",")
    data = [(lineparts[0],lineparts[1])]
infile.close()

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
print("Sequence "+filename[0:-4])
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

print("Extract per-residue representations (on CPU)")
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

res_embed=[]
token_temp=token_representations[0]
for i in range(1,len(token_temp)-1):
    temp_array=[]
    for j in range(len(token_temp[i])):
        temp_array.append(token_temp[i][j].item())
    res_embed.append(temp_array)
    # res_embed.append(token_representations[i,1:tokens_len - 1])
print("Generate per-sequence representations via averaging")
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# sequence = []
# for i, tokens_len in enumerate(batch_lens):
#     sequence.append(token_representations[i, 1 : tokens_len - 1].mean(0))



print("Writing outputs")

# opfile=open(base_dir+"met_var_sequences/"+filename[0:-4]+"_esm_embed.csv","w")
# opfile.write(filename[0:-4])
# seq_1=sequence[0].numpy()
# for j in seq_1:
#     opfile.write(","+str(j))
# opfile.write("\n")
# opfile.close()

opfile=open(base_dir+"met_var_sequences/"+filename[0:-4]+"_esm_residue_embed.csv","w")
opfile.write(filename[0:-4]+"\n")
for i in range(len(res_embed)):
    opfile.write(str(i+1)+",")
    opfile.write(",".join(str(x) for x in res_embed[i]))
    opfile.write("\n")
opfile.close()

