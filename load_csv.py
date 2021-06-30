import numpy as np
from os import path
from text import text_to_sequence

def load_csv(nm_csv, hps, utts=[], nms_data=[], split="|", sort_utt=True, check_org=True):
    # Ratio mel-frames per millisecond
    ratio_ms_data = hps.fe_data/1000.0

    with open(nm_csv, encoding='utf-8') as f:
        # Keep current audio-book chapter in memory
        nm_prec = ''

        # Loop on all lines of the .csv
        for line in f:
            # Split the line in ["chapter_name", start_segment (ms), end_segment (ms), "text"]
            fields = line.strip().split(split)
            nb_fields = len(fields)
            nm = fields[0]

            # Manage which mel-spectro file to look at
            if nm_prec != nm:
                nm_data = hps.prefix_data + nm + '.' + hps.extension_data
                if path.exists(nm_data):
                    # read file header to get dimensions
                    (lg_data, dim) = tuple(np.fromfile(nm_data, count=2, dtype=np.int32))

                    # Check if mel-spectro file has same dimensions than the current model used
                    if (dim != hps.dim_data):
                        lg = 0

                elif check_org:
                    # Ensure data is not considered if it has not been pre-processed
                    lg = 0
                    dim = 0
                    lg_data = 0
                    print('Error: {} not found'.format(nm_data),flush=True)
                else:
                    # ?
                    lg = lg_data = 1
                    dim = hps.dim_data

            # Manage current type of .csv
            if nb_fields >= 4:
                # Case 1: A segment of mel-spectro is used

                # Get first line in mel-spectro corresponding to current segment
                deb = int(ratio_ms_data*int(fields[1]))

                # Get length in mel-spectro corresponding to current segment
                if (hps.with_end_silence):
                    lg = int(ratio_ms_data * (int(fields[2]) + hps.end_silence_ms)) - deb
                else:
                    lg = int(ratio_ms_data*int(fields[2])) - deb

                # Index of text information
                i_txt = 3

            elif nb_fields >= 2:
                # Case 2: The entire mel-spectro is used
                deb = 0
                lg = lg_data

                # Index of text information
                i_txt = 1

            else:
                # Case 3: not enough fields, error
                deb = 0
                lg = 0
                i_txt = -1

            # Error conditions:
            # - number of mel spectro filters != from number of mel spectro in model
            # - wrong definition of mel-spectro
            # - end of segment exceeds mel-spectro duration
            # - length of segment exceeds max-length specified when running script
            if (dim != hps.dim_data) or (lg == 0) or (deb + lg > lg_data) or (lg > hps.lg_data_max):
                print('{}> Error data({},{},{}): {}, {}-{}, {}'.format(len(utts),nm,dim,lg_data,hps.dim_data,deb,lg,nb_fields), flush=True)

            else:
                # Get text information
                text_norm = text_to_sequence(fields[i_txt], ['basic_cleaners'])

                # Keep all seen chapters in nms_data, and get index of current one in i_nms
                try:
                    i_nms = nms_data.index(nm)
                except ValueError as e:
                    print('new datafile: {}'.format(nm))
                    i_nms = len(nms_data)
                    nms_data.append(nm)

                # Keep current audio-book chapter in memory
                nm_prec = nm

                # add processed csv line to output
                utts.append([i_nms, deb, lg, text_norm, len(text_norm)])

        # Print traces of processing
        nb_utts = len(utts)
        nb_nms = len(nms_data)
        print('{}: {} utts - {} datafiles'.format(nm_csv, nb_utts, nb_nms), flush=True)

        if sort_utt:
        	# sort by decreasing length
        	def takeLen_out(elem):
            		return elem[2]
        	# utterances finally sorted by length
        	utts.sort(key=takeLen_out,reverse = True)

    return(utts, nms_data)
