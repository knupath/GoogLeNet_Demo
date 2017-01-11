# L1/Cluster FPGA interface implementation
# dbarry
# modified to work with flitbridge
import socket
import struct
import time

import flit_utils as utils
from Header import Header1K
import PacketStream
import pyh1k

import spur
from spur.ssh import MissingHostKey

READ_RESPONSE_DRAIN_TIMEOUT = 3
MAX_PACKETS_OUT = 128

N_DEVICES = 1
PORTS_PER_DEVICE = 1



###########################################################################
class RemoteDut():

    #----------------------------------------------------------------------
    def __init__(self, name="DUT", host = 'localhost', base_port=31000, devices = 1, clusters = 1) :
    
        self._name = name
        self._open = False
        self._n_devices = devices
        self._n_clusters = clusters
        
        # We can run fun programs through ssh
        self._shell = spur.SshShell(
                            hostname=host,
                            username='root',
                            password='root',
                            missing_host_key=MissingHostKey.auto_add,
                        )
        self.host = host
        self.port = base_port
                            
        self.flitbridge_handshake()
        
        self._open = True


    #----------------------------------------------------------------------
    def __del__(self):
        """
        Destructor
        """
        self.close()
        
        
    #----------------------------------------------------------------------
    def localgo(self):
        """
        Assert clock to clusters defined in mask
        """
        mask = (2**self._n_clusters) - 1
        print "Checking localgo..."
        for device in xrange(self._n_devices):
            read = Header1K(
                    size=2,
                    opcode=0,
                    m=1,
                    device_id=device,
                    addr=0x20001404
                )
            response = Header1K(
                        header=read,
                        size=1,
                        opcode=1,
                        device_id=self.device_id
                    )
            self.write([[read.encode(), response.encode()]])
            localgo_response = self.listen(5 * 1000, n_flits=1)
            if len(localgo_response):
                localgo = localgo_response[0][2]
                print "localgo response received from device %d: %08X" % (device, localgo)
                if localgo != mask:
                    print "Setting localgo"
                    write = Header1K(
                                header=read,
                                size=1,
                                opcode=1,
                            )
                    self.write([[write.encode(), mask]])
            else:
                raise RuntimeError("No response received from Hermosa %d", device)
        
        
    #----------------------------------------------------------------------
    def flitbridge_handshake(self):
        device_id = 99
        default_request = 1<<20

        while True:
            self._dut_in_stream = PacketStream.PacketStream(
                                host = self.host,
                                port = self.port,
                                format = PacketStream.FORMAT_32BIT,
                                packet_format = PacketStream.PACKET_H1K,
                                wrap_packets = False,
                                blocking = False
                            )
            self._dut_in_stream._send(struct.pack('>I', device_id | default_request))

            # self._dut_in_stream._send(struct.pack('>I', 100))
            response = ''
            num_retries = 10
            
            while num_retries:
                response = self._dut_in_stream._read()
                if len(response):
                    break
                    
                time.sleep(0.5)
                
               
            if not len(response):
                raise RuntimeError("Flitbridge handshake timed out")
            
            response = struct.unpack('I', response)[0]
                
            if response != 0:
                print "Received non zero response %d" % response
                if response & 0x2000000:
                    print "Default port denied.  Removing request"
                    default_request = 0
                else:
                    print "Device %d denied" % device_id
                    device_id += 1
            else:
                print "Connected to flitbridge as %sdevice %d" % (
                            "default " if default_request else "", device_id)
                self.device_id = device_id
                break

            if device_id > 200:
                raise RuntimeError("Couldn't find available device id")


    def localgo(self):
        h = Header1K(
                size=1,
                opcode=1,
                m=1,
                addr=0x20001404
            )
            
        for device_id in xrange(self._n_devices):
            h.device_id = device_id
            self.write([[h.encode(), 0xFFFFFFFF]])  # Run all the clusters


    #----------------------------------------------------------------------
    def reset(self):
        """
        Soft resets the fpga device, clearing all fifos and memory.
        """
        result = self._shell.run(['python', 'soft-reset', '-t', 'localhost'], cwd=r'/usr/local/opt/knux/bin/')
        print "soft-reset output:"
        print result.output
        


    #----------------------------------------------------------------------
    def close(self):
        """
        Closes the device
        """
        if self._open:
            self._dut_in_stream.close()
            self._open = False

            
    #----------------------------------------------------------------------
    def init(self, packets):
        """
        Initializes memory and registers
        """
        self.write(packets)
        
        
    #----------------------------------------------------------------------
    def enable_flit_io(self):
        """
        Flit IO needs to be enabled on the supervisor after every reset
        """
        # DUT has two devices: 0 & 1
        devices = [0,1]


    #----------------------------------------------------------------------
    def read(self):
        """
        Reads whatever's in the FPGA outbox
        """
        return self._dut_in_stream.read()


    #----------------------------------------------------------------------
    def write(self, packets):
        """
        Writes a stream of packets
        """
        if not isinstance(packets, (list, tuple)):
            raise RuntimeError("device write() Invalid format for packets: %s" % repr(packets))

        packets = utils.convert_packets_64_to_32(packets)
    
        # Guy's code doesn't account for extra 32 bits of 0s on odd size
        # packets.  This is a workaround for now until he changes his code,
        # or we change our flit generation code.  I won't put this in the
        # convert function because it may break code on the FPGA.
        for i, p in enumerate(packets):
            # is size odd?
            if p[1] & 0x04000000:
                packets[i] = p[:-1]

        self._dut_in_stream.write(packets)
        
        # DEBUG testing loopback
        # Make sure we've connected this port before using.
        # self._dut_loopback_port.write(packets)



    #----------------------------------------------------------------------
    def listen(self, time_ms, n_flits=0):
        """
        Listens to a device for time_ms milliseconds and returns the output.
        If n_flits is specified, will stop listening after that many flits
        have arrived.
        """
        seconds = time_ms / 1000.0
        deadline = time.time() + seconds
        output = []

        while time.time() < deadline:

            out = self.read()
            if len(out):
                deadline = time.time() + seconds
                output.extend(out)
                
            if n_flits > 0 and len(output) >= n_flits:
                print "returning early from listen, {0} flits received".format(len(output))
                return output
                
            time.sleep(0.05)

        print "timing out from listen, {0}/{1} flits received".format(len(output), n_flits)
        
        return output
    

    #----------------------------------------------------------------------
    def exec_read_packets(self, packets, progress_callback=None):
        """
        Writes READ flits one by one to a device and returns the response as a
        stream of 32 bit ints without headers.
        """
        if not isinstance(packets, (list, tuple)):
            raise RuntimeError("device write() Invalid format for packets: %s" % repr(packets))

        packets = [list(p) for p in packets]
        
        # Add response index to packet.  This will break if we read more
        # than 4 gigawords from Hermosa at a time.
        read_length = 0
        for p in packets:
            p[1] |= read_length
            read_length += p[1] >> 58

        response = [0xBAD] * read_length

        inbound_index = 0
        outbound_count = 0
        inbound_count = len(packets)
        deadline = time.time() + READ_RESPONSE_DRAIN_TIMEOUT
        packets_out = 0
        last_packets_sent = 0

        while outbound_count < inbound_count:
            # Write a packet if there are any more
            if inbound_index < inbound_count and packets_out < MAX_PACKETS_OUT:
                # One packet at a time for now
                self.write([packets[inbound_index]])
                inbound_index += 1
                packets_out += 1

            # Check for responses
            responses = self.read()
            
            if len(responses):
                for packet in responses:
                    response[packet[0]:packet[0]+len(packet)-2] = packet[2:]
                outbound_count += len(responses)
                if progress_callback and (outbound_count - last_packets_sent) > 100:
                    progress_callback(outbound_count/float(inbound_count))
                    last_packets_sent = outbound_count
                packets_out -= len(responses)
                deadline = time.time() + READ_RESPONSE_DRAIN_TIMEOUT
            else:
                if time.time() > deadline:
                    print 'Timed out waiting for read response packets. %d of %d received' % (outbound_count, inbound_count)
                    break

        # Flatten response list, remove packet headers
        return response

