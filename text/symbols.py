""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

_cfrench = '[]§«»ÀÂÇÉÊÎÔàâæçèéêëîïôùûü¬~"' # gb: new symbols for turntaking & ldots, [] are for notes, " for new terms.

# Prepend "@" to phonetic symbols to ensure uniqueness (some are the same as lower letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_cfrench) + _arpabet + list('#') #GB add mark for emphasis