# Import Usefull libraries
import sys
import re
import numpy as np
import torch
import argparse
from os import path, system, chdir

from load_csv import load_csv
from text import text_to_sequence
from text.symbols import symbols
from model import Tacotron2, to_gpu

code_PAR=text_to_sequence('§',['basic_cleaners'])[0] # symbol for text spliting
code_POINT=text_to_sequence('.',['basic_cleaners'])[0] # symbol for end of utterance... to be replaced by end of paragraph at the end of each entry

from hparams import create_hparams
from scipy.io import savemat

hps = []
# exe = '/research/crissp/LPCNet-master/lpcnet_demo_NEB -synthesis'
exe = "python3 inference.py -f tacotron2.txt -w waveglow_NEB.pt -o . -s 0.6"

def synthesis(output_directory):
    if (hps.extension_data=='WAVEGLOW'):
        # Change repository to waveglow
        chdir('waveglow')

        # Save file to synthesize with waveglow
        cmd = 'ls ../{}/*.WAVEGLOW > tacotron2.txt'.format(output_directory); 
        system(cmd)
        print(cmd)

        # Synthesize with waveglow
        system(exe)
        print(exe)

        # Move .wav in appropriate file (same as .WAVEGLOW)
        cmd = 'mv *.wav ../{}/'.format(output_directory)
        system(cmd)
        print(cmd)

        # Back to current folder
        chdir('../')

# Save list of texts corresponding to generated utterances
def save_list_text(data_test, code_PAR, symbols):

    # Init variables
    list_split_text = []

    # Loop on utterances
    for i_syn in range(len(data_test)):

        # Load Utterance's text
        all_text_in=data_test[i_syn][3]

        # Get segmentation of text entry according to end of paragraph marks
        # parts = [0, endOfParagraphIndex1, endOfParagraphIndex2, ..., len(text_entry)]
        parts = [i for i,val in enumerate(all_text_in) if val==code_PAR]
        if all_text_in[len(all_text_in)-1]==code_PAR:
            parts[len(parts)-1]=len(all_text_in)
        else:
            parts=parts+[len(all_text_in)]

        if parts[0]:
            parts=[0]+parts

        # Initialize end punctuation replication at the beginning of the next split
        c_prec=all_text_in[0]

        # Loop on splits of text entry
        for ipart_txt in range(len(parts)-1):
            # text split prefixed by last character (punctuation) of previous split, no effects on first split
            all_text_in[parts[ipart_txt]]=c_prec
            c_prec=all_text_in[parts[ipart_txt+1]-1]

            # Get the text in the current split
            text_in=all_text_in[parts[ipart_txt]:parts[ipart_txt+1]]
            list_split_text.append(''.join([symbols[p] for p in text_in]))

    return list_split_text

nms_data=[]

# Process runs on GPU if available
device = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")

# Set args when running this script with cmd
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_directory', type=str,
                    help='directory to save generated files')
parser.add_argument('-p', '--prediction', required=False, action='store_true', default=False,
                    help='prediction instead of synthesis')
parser.add_argument('-g', '--ground_truth', required=False, action='store_true', default=False,
                    help='generate ground-truth parameter files')
parser.add_argument('--parameter_files', required=False, action='store_true', default=True,
                    help='generate parameter files')
parser.add_argument('-m', '--model', type=str, default='None', required=False,
                    help='Tacotron model')
parser.add_argument('-e', '--exe', type=str, default='/research/crissp/LPCNet-master/lpcnet_demo_DG -synthesis', required=True,
                    help='vocoder')
parser.add_argument('--hparams', type=str, required=False,
                    help='comma separated name=value pairs')

# Get hyper-parameters from args
args = parser.parse_args()
hps = create_hparams(args.hparams)
exe = args.exe
gen_pf = args.parameter_files

hps.mask_padding=False # only one file at a time!
model = Tacotron2(hps).to(device)

# Check if selected trained model exists, and load it
if path.exists(args.model):
    model_dict=torch.load(args.model, map_location='cpu')['state_dict']
    if hps.postnet==False:
        model_dict = {k: v for k, v in model_dict.items() if not re.match('^postnet',k)}
    model.load_state_dict(model_dict)
    print('Tacotron2 model "{}" loaded'.format(args.model))
else:
    print('Tacotron2 model "{}" not found'.format(args.model))
    sys.exit()

# Load CSV to synthesize
(data_test, nms_data)=load_csv(hps.nm_csv_test, hps, sort_utt=False, check_org=False)

# Diff sythesis / prediction based on args
if args.prediction==False:
    suffix='syn'
else:
    suffix='prd'

# if False: Solve memory issues when not in learning phase
torch.set_grad_enabled(False)

# Loop on utterances
for i_syn in range(len(data_test)):

    # ----------- Used for prediction / Ground Truth --------------

    # Concatenate generated file name
    nm = hps.prefix_data+nms_data[data_test[i_syn][0]]+'.'+hps.extension_data
    if path.exists(nm):
        # Load utterance's parameters
        deb = data_test[i_syn][1]
        lg_org = data_test[i_syn][2]

        # Load original mel-spectro section corresponding to the current utterance
        # float32 are encoded on 4 bytes => *4
        # +8 in offset corresponds to 2 float32 in header
        spe_org = np.memmap(nm,offset=8+(deb*hps.dim_data*4),dtype=np.float32,shape=(lg_org,hps.dim_data))

        # if ground-truth argument => generate ground-truth audio (= audio from ground-truth mel-spectro via vocoder)
        if args.ground_truth:
            nm_org='{}/{}_{:04d}_{}'.format(args.output_directory,nms_data[data_test[i_syn][0]],i_syn,'org')
            fp=open(nm_org+'.'+hps.extension_data,'wb')
            fp.write(np.asarray((lg_org,hps.dim_data),dtype=np.int32))
            fp.write(spe_org.copy(order='C'))
            fp.close()

            continue

            # # Do ground-truth synthesis
            # synthesis(nm_org)
    else:
        lg_org=0
    # ----------- END : Used for prediction / Ground Truth --------------

    # Load Utterance's text
    all_text_in=data_test[i_syn][3]

    if args.prediction==False:
        # Case 1: Synthesis
        # Generates output spectrum with model.inference

        # Initialize output spectrum
        spe_out_postnet=np.empty((0,hps.dim_data),dtype=np.float32)

        # Get segmentation of text entry according to end of paragraph marks
        # parts = [0, endOfParagraphIndex1, endOfParagraphIndex2, ..., len(text_entry)]
        parts = [i for i,val in enumerate(all_text_in) if val==code_PAR]
        if all_text_in[len(all_text_in)-1]==code_PAR:
          parts[len(parts)-1]=len(all_text_in)
        else:
          parts=parts+[len(all_text_in)]

        if parts[0]:
          parts=[0]+parts

        # Initialize end punctuation replication at the beginning of the next split
        c_prec=all_text_in[0]

        # Loop on splits of text entry
        for ipart_txt in range(len(parts)-1):
          if ipart_txt: # exclude first split
            # Copy 2 first lines at the bottom of the spectrum 0.3*hps.fe_data times ?
            for i in range(int(0.3*hps.fe_data)):
              spe_out_postnet=np.concatenate((spe_out_postnet,spe_out_postnet[0:1,:]))

          # text split prefixed by last character (punctuation) of previous split, no effects on first split
          all_text_in[parts[ipart_txt]]=c_prec
          c_prec=all_text_in[parts[ipart_txt+1]-1]

          # Get the text in the current split
          text_in=all_text_in[parts[ipart_txt]:parts[ipart_txt+1]]
          tensor_text_in=torch.Tensor(text_in)[None, :]
          tensor_text_in=to_gpu(tensor_text_in).long()

          # Synthesis of the split by the model
          (_, part_spe_out_postnet, _, part_alignement, part_encoder_out) = model.inference(tensor_text_in, hps.seed)

          # ----------- Save encoder embeddings of each chunk ----------------
          if hps.save_embeddings:
              # write split embedding in emb_mat
              emb_mat = part_encoder_out.cpu().data.numpy()[0].transpose()

              if ipart_txt:
                nm_emb='{}/{}_{}_emb_{}.mat'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix,ipart_txt) # Output filename
              else:
                nm_emb='{}/{}_{}_emb.mat'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix) # Output filename

              # save alignment in .mat format
              mdic = {"emb_mat": emb_mat}
              savemat(nm_emb, mdic)

          # ----------- Save Attention alignments of each chunk ----------------
          if hps.save_alignments:
              alignement_mat = part_alignement.cpu().data.numpy()
              alignement_mat = alignement_mat[0].transpose()

              if ipart_txt:
                nm_align='{}/{}_{}_align_{}.mat'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix,ipart_txt) # Output filename
              else:
                nm_align='{}/{}_{}_align.mat'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix) # Output filename

              # save alignment in .mat format
              mdic = {"align_mat": alignement_mat}
              savemat(nm_align, mdic)

          # Reshape the spectrum
          part_spe_out_postnet=part_spe_out_postnet.cpu().data.numpy()
          part_spe_out_postnet=part_spe_out_postnet.reshape(hps.dim_data,-1).transpose()
          # Concatenate this spectrum to the output
          spe_out_postnet=np.concatenate((spe_out_postnet,part_spe_out_postnet))
          print('synthesis of chunk {:d}[{:d}]: {}'.format(ipart_txt,part_spe_out_postnet.shape[0],''.join([symbols[p] for p in text_in])),flush=True)
    else:
        # Case 2: Prediction
        # Generates output spectrum with model.forward
        text_in=torch.Tensor(all_text_in)[None, :]; text_in=to_gpu(text_in).long()
        spe_tgt = spe_org.transpose(); spe_tgt = torch.Tensor(spe_tgt)[None, :]; spe_tgt=to_gpu(spe_tgt)
        lg_in=data_test[i_syn][4]
        (_, spe_out_postnet, _, part_alignement, part_encoder_out) = model.forward((text_in, [lg_in], spe_tgt, [lg_org]))
        
        # ----------- Save encoder embeddings of each chunk ----------------
        if hps.save_embeddings:
            # write split embedding in emb_mat
            emb_mat = part_encoder_out.cpu().data.numpy()[0].transpose()
            nm_emb='{}/{}_{}_emb.mat'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix) # Output filename

            # save alignment in .mat format
            mdic = {"emb_mat": emb_mat}
            savemat(nm_emb, mdic)

        # ----------- Save Attention alignments of each chunk ----------------
        if hps.save_alignments:
            alignement_mat = part_alignement.cpu().data.numpy()
            alignement_mat = alignement_mat[0].transpose()
            nm_align='{}/{}_{}_align.mat'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix) # Output filename

            # save alignment in .mat format
            mdic = {"align_mat": alignement_mat}
            savemat(nm_align, mdic)
        
        spe_out_postnet=spe_out_postnet.cpu().data.numpy()
        spe_out_postnet=spe_out_postnet.reshape(hps.dim_data,-1).transpose()

    # Initialize Output Variables    
    lg_out=spe_out_postnet.shape[0] # Output Length
    ch=''.join([symbols[p] for p in data_test[i_syn][3]])  #.replace('@',''), Output string
    nm_syn='{}/{}_{}'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix) # Output filename

    # If output file name already exists, generates alternative filenames : output_0, output_1...
    if path.exists(nm_syn+'.'+hps.extension_data):
        n=0;
        while n>=0:
            nm_syn='{}/{}_{}_{}'.format(args.output_directory,nms_data[data_test[i_syn][0]],suffix,n)
            if path.exists(nm_syn+'.'+hps.extension_data): n=n+1
            else: n=-1

    print('{}[{:d},{:d}]: {}'.format(nm_syn, lg_out, lg_org, ch), flush=True)

    # Copy of the spectrum in a file
    if args.parameter_files:
        fp=open(nm_syn+'.'+hps.extension_data,'wb')
        fp.write(np.asarray((lg_out,hps.dim_data),dtype=np.int32))
        fp.write(spe_out_postnet.copy(order='C'))
        fp.close()
        print('{}.{} created'.format(nm_syn,hps.extension_data), flush=True)

    # Spectrum to audio after each utt (not optimal)
    # synthesis(nm_syn)

if hps.save_embeddings or hps.save_alignments:
    # save text from utterances
    list_split_text = save_list_text(data_test, code_PAR, symbols)
    mdic = {"list_split_text":list_split_text}
    savemat('{}/list_split_text.mat'.format(args.output_directory), mdic)

# Spectrum => Audio all utterances at once
synthesis(args.output_directory)
