import socket
import struct

from Header import Header1K, Header2K


# Constants
RECEIVE_CHUNK_SIZE =    256*1024
LOCAL_STREAM_CONNECTION_PORT = 9001

FORMAT_64BIT = 'Q'
FORMAT_32BIT = 'I'

BYTES_32BIT = 4
BYTES_64BIT = 8

PACKET_H1K = 0
PACKET_H2K = 1

HEADER_SIZE = 8        # 8 byte header

# Error enumeration.  Maybe I'll add more of these later...
GENERAL_ERROR = range(1)
    


##############################################################################
class StreamException(Exception):
    pass
    
class SocketClosedException(Exception):
    pass
    
class RecvTimeoutException(Exception):
    pass

    

##############################################################################
class PacketStream(object):

    #-------------------------------------------------------------------------
    def __init__(
        self,
        host    = 'localhost',
        port    = LOCAL_STREAM_CONNECTION_PORT,
        listen    = False,
        format    = FORMAT_64BIT,
        blocking = False,
        packet_format = PACKET_H1K,    # Only referenced if wrap_packets is false
        wrap_packets = True
    ):

        self.socket = None
        self.host = host
        self.port = port
        self.listen = listen
        self.is_blocking = blocking
        self.format = '%d' + format
        self.format_size = BYTES_32BIT if format == FORMAT_32BIT else BYTES_64BIT
        self._wrap_packets = wrap_packets
        self._header_format = Header1K if packet_format == PACKET_H1K else Header2K
        
        self.buffer = ''

        self.open()


    #-------------------------------------------------------------------------
    def open(self):

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        if self.listen:

            sock.settimeout(None)
            sock.bind(('', self.port))
            sock.listen(0)
            
            s, addr = sock.accept()
            sock.close()
            sock = s
            
        else:

            try:
                sock.connect((self.host, self.port))
            except socket.timeout:
                raise StreamException("Connection timed out.  Destination: %s:%d" % 
                                        (self.host, self.port))

        sock.setblocking(self.is_blocking)
        self.socket = sock
        
        
    #-------------------------------------------------------------------------
    def get_client(self):
    
        return self.socket.getpeername()[0]
        
        
    #-------------------------------------------------------------------------
    def _send(self, bytes):
    
        n_bytes = len(bytes)
        sent_bytes = 0

        while sent_bytes < n_bytes:
            try:
                # send data starting from where we left off 
                sent_bytes += self.socket.send(bytes[sent_bytes:])  
            except socket.error as e:
                if e.errno != socket.errno.EWOULDBLOCK:
                    if (e.errno == socket.errno.EPIPE
                        or e.errno == socket.errno.ECONNRESET):
                        raise SocketClosedException()

                    raise e
        
        
    #-------------------------------------------------------------------------
    def write(self, packets):
        """
        Convert packets to binary data and send over socket connection.
        """
        data = self._pack_packets(packets)
        self._send(data)
        
        
    #-------------------------------------------------------------------------
    def _read(self):
        """
        Read socket
        """
        try:
            rec = self.socket.recv(RECEIVE_CHUNK_SIZE)
        except socket.timeout:
            # Timeout means we didn't receive any packets this time
            return ''
        except socket.error as e:
            if e.errno == socket.errno.EWOULDBLOCK:
                return ''
            elif (e.errno == socket.errno.EPIPE
                or e.errno == socket.errno.ECONNRESET):
                raise SocketClosedException()

            raise e
                
        if len(rec) == 0:
            raise SocketClosedException()
            
        return rec


    #-------------------------------------------------------------------------
    def read(self):
        """
        Take binary data from the socket and convert into packets.
        Append these to internal packet buffer.
        """
        self.buffer += self._read()
        return self._extract_packets()
        
        
    #-------------------------------------------------------------------------
    def error(self, errno):
        """
        Send an error to other side of pipe
        """
        self.write([errno])


    #-------------------------------------------------------------------------
    def close(self):

        self.socket.close()
        
        
    #-----------------------------------------------------------------------------
    def _pack_packets(self, packets):
        """
        Pack 2D list of packets into byte stream.
        """
        # Flatten list with size information
        if self._wrap_packets:
            flat = [i for packet in packets for i in ((len(packet),) + tuple(packet))]
        else:
            flat = [i for packet in packets for i in packet]
            
        packets = struct.pack(self.format % len(flat), *flat)
        
        return packets
        
        
    #-----------------------------------------------------------------------------
    def _extract_packets(self):
        """
        Extracts packets out of byte stream.  Returns 2D list of packets.
        """
        n_integers = len(self.buffer) / self.format_size
        packets = []

        # Need at least one value to extract size information
        if not n_integers:
            return packets

        integers = struct.unpack(
                    self.format % n_integers,
                    self.buffer[:n_integers * self.format_size]
                )
                
        # print "Packetstream received:", integers

        i = 0
        while i < n_integers:
        
            if self._wrap_packets:
                # Size info prepended to packet
                size = integers[i]
                start = i+1
            else:
                if self.format_size == BYTES_32BIT:
                
                    # Partial header
                    if not i+1 < n_integers:
                        break
                        
                    qword = integers[i+1] << 32
                else:
                    qword = integers[i]
                    
                h = self._header_format(qword)
                size = h.size + (HEADER_SIZE / self.format_size)
                
                start = i
                    
            end_index = start + size
            
            if end_index > n_integers:
                # Whole packet hasn't arrived on socket yet
                break
                
            packets.append(integers[start:end_index])
            
            i = end_index
            
        self.buffer = self.buffer[i * self.format_size:]

        return packets
