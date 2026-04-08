from datetime import datetime
from pylsl import StreamInfo, StreamOutlet, local_clock
from tmsi_dual_interface.tmsi_libraries.TMSiSDK import sample_data_server
import numpy as np
import time
import tmsi_dual_interface.tmsi_libraries.TMSiSDK.settings as settings




def start_stream():
    """
    Change parameters here
    """
    
    initialize()
    num_chan =  65
    global _consumer_list
    _consumer_list = ['1','2']

    file_writer_1 =  LSLWriter("FCR")
    file_writer_1.open( num_chan = num_chan, serial_num = _consumer_list[0])
    file_writer_2 =  LSLWriter("ECR")
    file_writer_2.open( num_chan = num_chan, serial_num = _consumer_list[1])
    print("producer started")
    while True:
        samp1 = list(np.random.normal(0,1,(num_chan)))
        samp2 = list(np.random.normal(0,1,(num_chan)))
        file_writer_1._outlet.push_sample(samp1)
        file_writer_2._outlet.push_sample(samp2)
        time.sleep(0.001)
    return None








def initialize():
    """Initializes the TMSi-SDK environment.
        This must be done once before starting using the SDK.
    """
    settings._initialize()

class LSLConsumer:
    '''
    Provides the .put() method expected by TMSiSDK.sample_data_server

    liblsl will handle the data buffer in a seperate thread. Since liblsl can
    bypass the global interpreter lock and python can't, and lsl uses faster
    compiled code, it's better to offload this than to create our own thread.
    '''

    def __init__(self, lsl_outlet):
        self._outlet = lsl_outlet

    def put(self, sd):
        '''
        Pushes sample data to pylsl outlet, which handles the data buffer

        sd (TMSiSDK.sample_data.SampleData): provided by the sample data server
        '''
        try:
            # split into list of arrays for each sampling event
            signals = [sd.samples[i*sd.num_samples_per_sample_set : \
                                (i+1)*sd.num_samples_per_sample_set] \
                                    for i in range(sd.num_sample_sets)]
            # and push to LSL
            self._outlet.push_chunk(signals, local_clock())
        except:
            raise "Write error"


class LSLWriter:
    '''
    A drop-in replacement for a TSMiSDK filewriter object
    that streams data to labstreaminglayer
    '''

    def __init__(self, stream_name = ''):

        self._name = stream_name if stream_name else 'tmsi'
        self._consumer = None
        self._date = None
        self._outlet = None


    def open(self, sample_rate = 4000, num_chan = 64, serial_num = '1'):
        '''
        Input is an open TMSiSDK device object
        '''

        print("LSLWriter-open")
        # self.device = device

        try:
            self._date = datetime.now()
            self._sample_rate = sample_rate
            # self._sample_rate = device.config.sample_rate
            self._num_channels = num_chan
            # self._num_channels = len(device.channels)

            self._num_sample_sets_per_sample_data_block = int(self._sample_rate * 0.15)
            size_one_sample_set = num_chan * 4
            if ((self._num_sample_sets_per_sample_data_block * size_one_sample_set) > 64000):
                self._num_sample_sets_per_sample_data_block = int(64000 / size_one_sample_set)

            # provide LSL with metadata
            info = StreamInfo(
                self._name,
                'EEG',
                self._num_channels,
                self._sample_rate,
                'float32',
                'tmsi-' + str(serial_num), 
                ) 
            chns = info.desc().append_child("channels")
            ch_name = "mock chan"
            unit_name = "mock unit"
            for idx in range(num_chan): # active channels
                chn = chns.append_child("channel")
                chn.append_child_value("label", ch_name)
                chn.append_child_value("index", str(idx))
                chn.append_child_value("unit", unit_name)
                chn.append_child_value("type", 'EEG')
            info.desc().append_child_value("manufacturer", "TMSi")
            sync = info.desc().append_child("synchronization")
            sync.append_child_value("offset_mean", str(0.0335)) # measured while dock/usb connected
            sync.append_child_value("offset_std", str(0.0008)) # jitter AFTER jitter correction by pyxdf

            # start sampling data and pushing to LSL
            self._outlet = StreamOutlet(info, self._num_sample_sets_per_sample_data_block)
            self._consumer = LSLConsumer(self._outlet)
            sample_data_server.registerConsumer(serial_num, self._consumer)

        except:
            raise "running issues"


    def close(self):

        print("LSLWriter-close")
        sample_data_server.unregisterConsumer(self.device.id, self._consumer)
        # let garbage collector take care of destroying LSL outlet
        self._consumer = None
        self._outlet = None



if __name__ == "__main__":
    start_stream()