import numpy as np

class SpikeEncoder:
    def __init__(self):
        pass
    
    def evolve(self, sig_in):
        raise NotImplementedError("Method evolve not implemented")
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    

class ZeroCrossingSpikeEncoder(SpikeEncoder):
    def __init__(self, fs=16_000, robust_width=1, bipolar=False):
        self.fs = fs
        self.robust_width = robust_width
        self.bipolar = bipolar

    def zero_crossing(self, sig_in):
        spikes = np.zeros_like(sig_in).T

        for chan, sig_chan in enumerate(sig_in.T):
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(sig_chan)))[0]
            #zero_crossings = np.where(np.diff(np.signbit(sig_chan.astype(np.float32))))[0]
            spikes[chan, zero_crossings] = 1

            if self.bipolar:
                # Also include negative to positive crossings
                zero_crossings_neg = np.where(np.diff(np.signbit(-sig_chan)))[0]
                spikes[chan, zero_crossings_neg] = -1

        return spikes.T