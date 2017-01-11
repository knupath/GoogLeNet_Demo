from Header import Header1K

# Packet opcodes
FLIT_OP_READ = 0
FLIT_OP_WRITE = 1
FLIT_OP_WRDEC = 3

# Memory units
FLT_MEM_UNIT_DEVICE = 1
FLT_MEM_UNIT_CLUSTER = 2
FLT_MEM_UNIT_TDSP = 3

# Addresses
CCR_ADDR_FCR = 589824
TDSP_ADDR_CMD = 12288
TDSP_ADDR_REGFILE_START = 0
TDSP_ADDR_MBOX_START = 192
TDSP_ADDR_SPEC_REG_START = 0x3000
TDSP_ADDR_SPEC_REG_END = 0x303E
TDSP_ADDR_PROGMEM = 0x1000
TDSP_PROGMEM_SIZE = 0x800

CLUSTER_EMEM_START = 0
CLUSTER_SMEM_START = 0x80000
CLUSTER_CCR_START = 0x90000
CCR_ADDR_END = 0x90023
CCR_ADDR_CER    = 0x0090002

CLUSTERS_PER_DEVICE = 32
TDSPS_PER_CLUSTER = 8

H1000_PKT_HDR_LWRDS = 2


def convert_64_to_32(val):
    return ((val >> 32) & 0xFFFFFFFF, val & 0xFFFFFFFF)

def convert_vector64_to_vector32(vec):
    return tuple(((i >> (32*j)) & 0xFFFFFFFF) for i in vec for j in xrange(2))
    
def convert_vector32_to_vector64(vec):

    if len(vec) % 2:
        vec = vec + [0]
        
    # Note no size checking on 32 bit integers
    return tuple(vec[i] | (vec[i+1] << 32) for i in xrange(0, len(vec), 2))
    
def read_hexfile_as_32(file):

    f = open(file)
    lines = f.read().splitlines()
    f.close()
    
    hexfile = []
    
    for line in lines:
        try:
            line = line[:line.index(';')]
        except ValueError as e:
            # No comment on line
            pass
            
        line = line.strip()
        
        if not line:
            continue
            
        hexfile.append(int(line, 16))
        
    return hexfile
    
    
def read_hexfile_as_64(file):

    return convert_vector32_to_vector64(read_hexfile_as_32(file))
    

def Uint32Vector(vec):
    return vec
    
    
def make_read_packets(address, n_items, device_id, cluster_id, tdsp_id = None):

    m = 2 if tdsp_id is None else 3
    
    packets = []

    while n_items > 0:
    
        if n_items > 32:
            size = 32
        else:
            size = n_items
            
        n_items -= size
        
        read = Header1K(
                size    = 2,
                opcode    = 0,
                m        = m,
                device_id    = device_id,
                cluster_id    = cluster_id,
                tdsp_id        = tdsp_id,
                addr        = address,
            )
                
        ret = Header1K(
                size    = size,
                opcode    = 1,
                m        = 1,
                device_id    = 99,
            )
            
        packets.append([read.encode(), ret.encode()])
            
        address += size
        
    return packets
    
    
def make_write_packets(address, data, device_id, cluster_id, tdsp_id = None):

    m = 2 if tdsp_id is None else 3
    
    packets = []
    n_items = len(data)
    index = 0

    while n_items > 0:
    
        if n_items > 32:
            size = 32
        else:
            size = n_items
            
        n_items -= size
        
        write = Header1K(
                size    = size,
                opcode    = 1,
                m        = m,
                device_id    = device_id,
                cluster_id    = cluster_id,
                tdsp_id        = tdsp_id,
                addr        = address + index,
            )
            
        packets.append((write.encode(),) + convert_vector32_to_vector64(data[index:index+size]))
            
        index += size
        
    return packets
        
    
class FlitPacketHeader(object):

    def __init__(self, word = 0):
        self._header = Header1K(word)
        
    def get_size(self):
        return self._header.size
        
    def get_opcode(self):
        return self._header.opcode
        
    def get_m(self):
        return self._header.m
        
    def get_device_id(self):
        return self._header.device_id
        
    def get_cluster_id(self):
        return self._header.cluster_id
        
    def get_tdsp_id(self):
        return self._header.tdsp_id
        
    def get_physical_address(self):
        return self._header.addr
        
    def set_size(self, value):
        self._header.size = value
        
    def set_opcode(self, value):
        self._header.opcode = value
        
    def set_m(self, value):
        self._header.m = value
        
    def set_v(self, value):
        self._header.v = value
        
    def set_device_id(self, value):
        self._header.device_id = value
        
    def set_cluster_id(self, value):
        self._header.cluster_id = value
        
    def set_tdsp_id(self, value):
        self._header.tdsp_id = value
        
    def set_physical_address(self, value):
        self._header.addr = value
        
    def set_physical_addr(self, value):
        self._header.addr = value
        
    def set_event_flags(self, value):
        self._header.evfm = value
        
    def encode(self):
        return self._header.encode()
        
    @staticmethod
    def extract_size(word):
        return Header1K(word).size
        
    @staticmethod
    def extract_opcode(word):
        return Header1K(word).opcode