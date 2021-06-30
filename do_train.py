import numpy as np
import numpy.ma as ma
import torch
from torch import nn
from torch.utils.data.sampler import Sampler
import argparse
from os import path
from load_csv import load_csv
from text.symbols import symbols
from model import Tacotron2
from hparams import create_hparams

hps = []
nms_data=[]

# Process runs on GPU if available
device = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")

# This function is used for training the network with equivalent length utterance (based on audio)
# Utterances are ordered by increasing size, and sample groups are made of randomly chosen succesive utterances
class OrderedSampler(Sampler):

    def __init__(self, model, train_data, batch_size, drop_last=True):
        self.model = model
        self.train_data = train_data
        self.batch_size = batch_size
        self.nb_utts = len(data_train)
        self.drop_last = drop_last

    def __iter__(self):
        lst=np.arange(self.nb_utts)
        while len(lst)>=self.batch_size:
            i_first=np.random.randint(len(lst))
            ind=np.argsort(abs(lst-i_first))
            batch=lst[ind[0:self.batch_size]]
            print('{}/{}'.format(len(batch),len(lst)))
            lst=np.delete(lst,ind[0:self.batch_size])
            yield batch
        print('LAST:= {}/{} {}'.format(len(batch),len(lst),self.drop_last))
        if len(lst) > 0 and not self.drop_last:
            yield lst

    def __len__(self):
        return self.nb_utts

# Classical random selection of training utterances  
class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch
        if len(batch) > 0 and not self.drop_last:
           yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size

def get_mask_from_lengths(lengths):
    max_lengths = max(lengths); nb_lengths=len(lengths)
    ids=np.arange(0, max_lengths)
    mask=(ids<np.reshape(lengths,(nb_lengths,1)))
    mask=torch.from_numpy(mask).cuda()
    return mask

def collate_batch(batch):
    lg_batch = len(batch)
    if (hps.verbose_logs):
        print('collate_batch {}'.format(lg_batch))

    # sort by increasing input length
    def takeLen_in(elem):
        return elem[4]
    batch.sort(key=takeLen_in,reverse = True)
    lg_in = [item[4] for item in batch]; max_lg_in = max(lg_in)
    text_in = torch.zeros([lg_batch, max_lg_in], dtype=torch.long)
    lg_tgt = [item[2] for item in batch]; max_lg_tgt = max(lg_tgt)
    spe_tgt = torch.zeros([lg_batch, hps.dim_data, max_lg_tgt],dtype=torch.float32)
    gate_tgt = torch.zeros([lg_batch, max_lg_tgt],dtype=torch.float32)
    for i_batch in range(len(batch)):
        text_in[i_batch,:batch[i_batch][4]] = torch.Tensor(batch[i_batch][3])
        nm = hps.prefix_data+nms_data[batch[i_batch][0]]+'.'+hps.extension_data; deb = batch[i_batch][1]; lg = batch[i_batch][2]
        spe = np.memmap(nm,offset=8+(deb*hps.dim_data*4),dtype=np.float32,shape=(lg,hps.dim_data)).transpose()
        spe_tgt[i_batch,:, :lg] = torch.Tensor(spe)

        if (hps.with_end_silence):
            gate_tgt[i_batch,lg-int(2+hps.fe_data*hps.end_silence_ms/1000.0):] = 1 # min 2 frames at the end
        else:
            gate_tgt[i_batch,lg-2:] = 1 # 2 frames at the end

        ch=''.join([symbols[p] for p in batch[i_batch][3]]).replace('@','')
        if (hps.verbose_logs):
            print('{}: {:3d} -> {:4d} from {:6d}: {}'.format(nms_data[batch[i_batch][0]],len(batch[i_batch][3]),lg,deb,ch), flush=True)

    i_nm = [[item[0],item[1]] for item in batch]
    return [text_in, lg_in, spe_tgt, gate_tgt, lg_tgt, i_nm] # spe_tgt used for teacher forcing

def warm_start_model(nm_mod, model, ignore_layers):
    print("Warm starting model '{}'".format(nm_mod), flush=True)
    checkpoint_dict = torch.load(nm_mod, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']

    # Reload character embeddings
    car_embeddings=model_dict.get('embedding.weight'); nb_car=car_embeddings.shape[0]
    # add unseen characters
    if (nb_car <len(symbols)): # extra characters have been added
        car_embeddings=torch.cat((car_embeddings,torch.zeros(len(symbols)-nb_car,hps.symbols_embedding_dim)))
        model_dict.update({('embedding.weight', car_embeddings)})
        print('{} symbols added'.format(len(symbols)-nb_car))

    dim=model_dict.get('decoder.linear_projection.linear_layer.weight').shape[0]
    if (dim!=hps.dim_data):
        hps.postnet=False # get rid of the postnet for transfer learning
        lst = ['decoder.prenet.layers.0.linear_layer.weight','decoder.linear_projection.linear_layer.weight','decoder.linear_projection.linear_layer.bias','postnet.convolutions.0.0.conv.weight','postnet.convolutions.4.0.conv.weight','postnet.convolutions.4.0.conv.bias','postnet.convolutions.4.1.weight','postnet.convolutions.4.1.bias','postnet.convolutions.4.1.running_mean','postnet.convolutions.4.1.running_var'] # do not consider free parameters that differ in dim
        if ignore_layers is not None:
            ignore_layers = ignore_layers+lst
        else:
            ignore_layers = lst
        print('ignore_layers {}'.format(ignore_layers))
    if ignore_layers is not None:
        model_dict = {k: v for k, v in model_dict.items()
            if k not in ignore_layers}
        for k, v in model_dict.items():
            print(k)
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict,strict=False)
    print("Model '{}' loaded".format(nm_mod), flush=True)
    return model

# Save intermediate results during training
def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

# Perform training if is_train = True, validation if is_train = False
def process(model, is_train, device, loader, optimizer, epoch, by_utt=False, nt=0, T_Perf=np.empty((0,3))):
    if is_train:
        #Case 1: Training
        model.train()
        phase = 'train'

        # if False: Solve memory issues when not in learning phase
        torch.set_grad_enabled(True)
    else:
        # Case 2: Validation
        model.eval()
        phase = 'validation'

        # if False: Solve memory issues when not in learning phase
        torch.set_grad_enabled(False)

    print('{} on {:3d} sequences'.format(phase,len(loader.dataset)))

    # Initialize Variables
    loss_spe_postnet = 0.0
    mean_loss = 0.0
    mean_err_lg = 0.0
    mean_gate_loss = 0.0
    mean_spe_loss = 0.0
    mean_spe_postnet_loss = 0.0
    lg_tgt = np.empty((0,1),dtype='int16')
    lg_prd = np.empty((0,1),dtype='int16')

    # Loop on all batches
    for i_batch, batch in enumerate(loader):
        # Get batch length (= number of utterances in batch)
        lg_batch = len(batch[0])

        if is_train:
            # Clear gradient
            optimizer.zero_grad()

        # Get current batch infos (inputs + targets)
        (T_in, T_tgt, i_nm) = model.parse_batch(batch)

        # Predict outputs of current batch
        (spe_out, spe_out_postnet, gate_out, _, _) = model(T_in)
        (spe_tgt, gate_tgt) = T_tgt

        # Find "Stop tokens"
        ind = np.where(gate_out.cpu() > hps.gate_threshold)
        ind_ok = np.unique(ind[0], return_index=True)[1]

        # Get predictions length
        lg = hps.lg_data_max * np.ones(lg_batch, dtype='int16')
        lg[ind[0][ind_ok]] = ind[1][ind_ok] + 2

        # Computes error length between prediction and target
        lg_prd = np.append(lg_prd,lg)
        if (hps.with_end_silence):
            lg_tgt=np.append(lg_tgt,np.array(batch[4])-int(hps.fe_data*hps.end_silence_ms/1000.0))
            err_lg=np.mean(abs(lg-batch[4]))-int(hps.fe_data*hps.end_silence_ms/1000.0)
        else:
            lg_tgt = np.append(lg_tgt,batch[4])
            err_lg = np.mean(abs(lg-batch[4]))

        if by_utt:
            # ----------- Case: During Validation ------------

            # Computes error between prediction and target (Mean Square Error used)
            loss_spe = nn.MSELoss(reduction='none')(spe_out, spe_tgt)
            NN_spe = loss_spe.cpu().detach().numpy()
            NN_spe = NN_spe.mean(axis=1)
            
            # with mask
            # Computes mask from target length
            mask = ~get_mask_from_lengths(T_in[3]).cpu().detach().numpy()
            NN_spe = ma.masked_array(NN_spe, mask)

            # Computes Binary-Cross-Entropy Loss on "stop token" prediction
            loss_gate = hps.loss_gate_weight * nn.BCEWithLogitsLoss(reduction='none')(gate_out, gate_tgt)
            NN_gate = loss_gate.cpu().detach().numpy()
            NN_gate = ma.masked_array(NN_gate,mask)

            # Global Loss
            g_loss_per_utt = NN_spe.mean(axis=1).data + NN_gate.mean(axis=1).data
            g_loss = NN_spe.mean() + NN_gate.mean()

            # Loss are computed before and after postnet
            if hps.postnet:
                loss_spe_postnet = nn.MSELoss(reduction='none')(spe_out_postnet, spe_tgt)
                NN_spe_postnet = loss_spe_postnet.cpu().detach().numpy()
                NN_spe_postnet = NN_spe_postnet.mean(axis=1)
                NN_spe_postnet = ma.masked_array(NN_spe_postnet,mask)
                
                g_loss_per_utt += NN_spe_postnet.mean(axis=1).data
                g_loss += NN_spe_postnet.mean()

            # Concatenate output result
            Perf = np.append(np.array(i_nm), np.array(g_loss_per_utt).reshape((lg_batch,1)), axis=1)
            T_Perf = np.append(T_Perf,Perf,axis=0)

        else:
            # ----------- Case: During Training ------------

            # Computes error between prediction and target (Mean Square Error used)
           loss_spe = nn.MSELoss()(spe_out, spe_tgt)
           # Computes Binary-Cross-Entropy Loss on "stop token" prediction
           loss_gate = hps.loss_gate_weight * nn.BCEWithLogitsLoss()(gate_out, gate_tgt)
           
           # Global loss
           loss = loss_spe + loss_gate

           # Loss are computed before and after postnet
           if hps.postnet:
                loss_spe_postnet = nn.MSELoss()(spe_out_postnet, spe_tgt)
                loss  += loss_spe_postnet

           g_loss = loss.item()

        # Loop on utterances of current batch
        for i_b in range(lg_batch):
            l_org = batch[1][i_b]
            ch = ''.join([symbols[p] for p in batch[0][i_b][0:l_org]]) #.replace('@','')
            
            if (hps.verbose_logs):
                print('{} at {}, {} "{}" -> {}'.format(nms_data[batch[5][i_b][0]], batch[5][i_b][1], batch[4][i_b], ch, lg[i_b]), end = '')

            if by_utt:
              print(', LOSS={:.3}'.format(g_loss_per_utt[i_b]),end='')

        if is_train:
            # Back-propagation
            loss.backward()
            optimizer.step()

        nt += lg_batch
        print('{} Batch: {:3d}/{:6d} ({:.2f}%)]\tLoss: {:.4f}, Err_lg: {:.2f}, Gate Loss: {:.3f}, Spe Loss: {:.3f}, Spe Postnet Loss: {:.3f}'.format(phase, nt, len(loader.dataset), 100.*nt/len(loader.dataset), g_loss, err_lg, loss_gate/hps.loss_gate_weight, loss_spe, loss_spe_postnet), flush=True)
        mean_loss += g_loss
        mean_err_lg += err_lg

        mean_gate_loss += loss_gate
        mean_spe_loss += loss_spe
        if hps.postnet:
            mean_spe_postnet_loss += loss_spe_postnet

    # Mean loss is mean over batches
    mean_loss /= (i_batch + 1)
    mean_err_lg /= (i_batch + 1)
    mean_gate_loss /= (i_batch + 1)
    mean_spe_loss /= (i_batch + 1)
    mean_spe_postnet_loss /= (i_batch + 1)

    # nb of predictions that do not exceed lg_max (= number of ending synthesis)
    pct_ok = (100.0*sum(lg_prd < hps.lg_data_max)) / len(lg_prd)
    print('{} Epoch: {} Mean Loss: {:.3f}, Mean Err_lg: {:.2f},{:.2f}%, Mean Gate Loss: {:.3f}, Mean Spe Loss: {:.3f}, Mean Spe Postnet Loss: {:.3f}'.format(phase, epoch, mean_loss, mean_err_lg, pct_ok, mean_gate_loss/hps.loss_gate_weight, mean_spe_loss, mean_spe_postnet_loss), flush=True)
    return (nt, T_Perf)

#############################################################################
###########                 Main Script                 #####################
#############################################################################

# Set args when running this script with cmd   
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_directory', type=str, default='_out',
                    help='directory to save checkpoints')
parser.add_argument('-c', '--pre_trained', type=str, default=None,
                    required=False, help='pre-trained model')
parser.add_argument('--warm_start', action='store_true',
                    help='load model weights only, ignore specified layers')
parser.add_argument('--hparams', type=str,
                    required=False, help='comma separated name=value pairs')
parser.add_argument('--freeze_encoder', type=str, default=False,
                    help='freeze encoder for transfer learning')
args = parser.parse_args()
hps = create_hparams(args.hparams)

model = Tacotron2(hps).to(device)

# If warm-start: need to specify a valid path for pre-trained model
if path.exists(args.pre_trained) and args.warm_start:
    model = warm_start_model(args.pre_trained, model, hps.ignore_layers)

# Case Freeze Encoder
if args.freeze_encoder:
    model.freeze_encoder()

# Load training csv
(data_train, nms_data) = load_csv(hps.nm_csv_train, hps)

# Manage types of sampling for batch selection: (BatchSampler or OrderedSampler)
sampler_train = OrderedSampler(model, data_train, hps.batch_size, drop_last=False)
train_loader = torch.utils.data.DataLoader(data_train, batch_sampler=sampler_train, collate_fn=collate_batch, num_workers=0)

# Load test csv
# nms_data are concatenated with nms_data from training csv
(data_test, nms_data) = load_csv(hps.nm_csv_test, hps, utts=[], nms_data=nms_data)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=hps.batch_size, drop_last=False, shuffle=False, collate_fn=collate_batch, num_workers=0)

nb_training_iter=0
nbr_epoch = hps.offset_epoch
for i_epoch in range (hps.nb_epochs):
    nbr_epoch += 1
    # Check is postnet is activated for this epoch
    if hps.start_without_postnet:
        if nbr_epoch >= hps.epoch_start_postnet:
            hps.postnet = True
        else:
            hps.postnet = False

    # Calculate learning rate
    lr = max(hps.learning_rate_min, hps.learning_rate_max*min(1,np.exp(-hps.dr*(nbr_epoch-hps.epoch_start)/hps.epoch_start)))
    # Manage optimization
    # optimizer = optim.SGD(model.parameters(), lr=hps.learning_rate, momentum=hps.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=hps.weight_decay)

    # Training
    (nt, _) = process(model, True, device, train_loader, optimizer, nbr_epoch)
    nb_training_iter += nt

    # Validation performed after each epoch
    # process(model, False, device, train_loader, [], nbr_epoch)
    process(model, False, device, test_loader, [], nbr_epoch)

    # Save intermediate results
    if nbr_epoch%hps.checkpoint_freq == 0:
        nm_mod = path.join(args.output_directory, "tacotron2_{}_{}".format(hps.extension_data,nbr_epoch))
        save_checkpoint(model, optimizer, lr, nb_training_iter, nm_mod)

# Validation performed at the end of the process
print('Final performances:')
# is_train = false, no optimizer, i_epoch = 0
(nt_train, T_Perf_train) = process(model, False, device, train_loader, [], 0, by_utt=False)
(nt_test, T_Perf_test) = process(model, False, device, test_loader, [], 0, by_utt=False)