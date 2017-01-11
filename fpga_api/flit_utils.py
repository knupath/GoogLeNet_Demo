
import pyh1k as h1ksim
	
from Header import Header1K

RETURN_DEVICE_ID = 99

FLAG_TEXT_CONCAT_STRING = ', '


def pattern_debug(num_values, signature = 0xDB000000) :

	return range(signature, signature + num_values)

def pattern_zeros(num_values):

	return [0] * num_values

get_fill_pattern = pattern_debug


def make_readback_packets(packets):
	"""
	Create read packets to readback what was written by parameter packets.
	packets is 64 bit packet vector.
	"""
	readback_packets = []
	
	for i, write in enumerate(packets):
	
		re = Header1K(
				header = write[0],
				m = 1,
				device_id = RETURN_DEVICE_ID,
				addr = i		# Keep track of response order
			)
			
		rd = Header1K(
				header = write[0],
				size = 2,
				opcode = h1ksim.FLIT_OP_READ,
			)
			
		readback_packets.append([rd.encode(), re.encode()])
		
	return readback_packets


def output_packets_as_stream(packets, filename, strip_headers = True) :
	""" Outputs a list of packets to a file. """

	strip = 2 if strip_headers else 0

	f = open(filename, 'w')
	for packet in packets :
		for value in packet[strip:] :
			f.write("%08X\n" % value)
	f.close()

def count_kickoff_packets(packets):
	"""
	Counts the number of packets sent after program memory is filled.
	"""
	count = 0
	start_index = len(packets) - 1

	# Loop from end of array to beginning
	for i in xrange(start_index, -1, -1):
		header = h1ksim.FlitPacketHeader(packets[i][0])
		m = header.get_m()
		address = header.get_physical_addr()

		if not ((m == h1ksim.FLT_MEM_UNIT_CLUSTER and address == h1ksim.CCR_ADDR_FCR) \
			or (m == h1ksim.FLT_MEM_UNIT_TDSP and address == h1ksim.TDSP_ADDR_CMD)):
				break

		count += 1

	return count


def output_stream(stream, filename) :
	""" Outputs a packet stream for debugging. """

	f = open(filename, 'w')
	for value in stream :
		f.write(NUMBER_OUTPUT_FORMAT_STRING + "\n" % value)
	f.close()


def print_stream(stream) :

	for x in stream :
		print "%08x" % x


def replace_kickoffs_with_stepoffs(packets):
	"""
	Changes kickoff packets to stepoff packets.
	Will change the flit stream passed in so be careful.
	"""
	for packet in packets:
		header = h1ksim.FlitPacketHeader(packet[0])

		if header.get_physical_address() == h1ksim.TDSP_ADDR_CMD and header.get_m() == h1ksim.FLT_MEM_UNIT_TDSP:
			packet[1] = h1ksim.TDSP_REQUEST_STEP
			
			
def strip_kickoffs(packets):
	"""
	Remove kickoffs from flit stream and return them in a separate list
	"""
	kickoffs = []
	start_index = len(packets) - 1
	
	# Move backwards through the list of packets so we can remove items
	for i in xrange(start_index, -1, -1):
		header = h1ksim.FlitPacketHeader(packets[i][0])
		m = header.get_m()
		address = header.get_physical_addr()

		if not ((m == h1ksim.FLT_MEM_UNIT_CLUSTER and address == h1ksim.CCR_ADDR_FCR) \
			or (m == h1ksim.FLT_MEM_UNIT_TDSP and address == h1ksim.TDSP_ADDR_CMD)):
				continue
				
		kickoffs.append(packets[i])
		del packets[i]
		
	return kickoffs


def verify_packet_headers(packets) :
	"""
	Verify the headers of a list of packets.
	"""

	for packet in packets :
		try :
			header = h1ksim.FlitPacketHeader(packet[0])
		except:
			return False

	return True


def convert_to_stream(pkts, strip_headers = True) :
	""" Converts a list of packet tuples to a stream of ints """

	strip = 2 if strip_headers else 0

	stream = []

	for pkt in pkts :
		stream.extend(pkt[strip:])

	return stream


def convert_packets_64_to_32(packets) :
	"""Convert a list of 64 bit packet tuples to a 32 bit stream """

	stream = []

	for packet in packets :
		stream.append(h1ksim.convert_vector64_to_vector32(packet))

	return stream


def convert_packets_32_to_64(packets) :
	"""Convert a list of 32 bit packet tuples to 64 bit packets"""

	stream = []

	for packet in packets:
		if len(packet) % 2:
			packet.append(0)
		stream.append(h1ksim.convert_vector32_to_vector64(packet))

	return stream


def set_device_id_to(packets, device_id):
	'''
	Set destination and read return device id to device_id.
	Used when communicating with USB FPGA using get_flit and send_flit.
	64 bit packets required.
	'''
	mask = ~(h1ksim.FLT_PHY_MASK_DEV)
	device_id <<= h1ksim.FLT_PHY_SHIFT_DEV

	new_packets = []

	for p in packets:
		new_packet = list(p)
		new_packet[0] = (p[0] & mask) | device_id

		# Read return will also be set
		if h1ksim.FlitPacketHeader.extract_opcode(p[0]) == h1ksim.FLIT_OP_READ:
			new_packet[1] = p[1] & mask | device_id

		new_packets.append(new_packet)

	return new_packets


def create_packet_tuples(stream) :
	""" Turn stream of 64 bit ints into list of packet tuples """

	packets = []

	i = 0
	while i < len(stream) :
		size = h1ksim.FlitPacketHeader.extract_size(stream[i])

		#Get number of 64 bit ints this packet spans.
		size = ((size + 1) / 2) + 1

		packets.append(tuple(stream[i:i + size]))

		i += size

	return packets


def create_packet_tuples_32(stream) :
	""" Convert stream of 32 bit ints to 32 bit packets with 64 bit header """

	packets = []

	stream_length = len(stream)

	i = 0
	while i < stream_length :
		h64 = h1ksim.convert_32_to_64(header[1], header[0])

		try :
			#Make sure this is actually a flit header
			fph = h1ksim.FlitPacketHeader(h64)
			size = fph.get_size() + h1ksim.H1000_PKT_HDR_LWRDS
			packets.append(h64 + stream[i:i + size])

			i += size
		except :
			print "Invalid header found in stream: %016X" % h64
			i += 2



def make_cluster_mem_init_packets(address_start, device_id, cluster_id, size) :

	filler_data = get_fill_pattern(size, 0xDBE00000 + address_start)

	return h1ksim.make_cluster_mem_write_packets(
		h1ksim.Uint32Vector(filler_data),
		address_start,
		cluster_id,
		device_id)


def make_emem_init_packets(device_id, cluster_id, size) :

	filler_data = get_fill_pattern(size, 0xDBE00000)

	return h1ksim.make_cluster_mem_write_packets(
		h1ksim.Uint32Vector(filler_data),
		h1ksim.CLUSTER_EMEM_START,
		cluster_id,
		device_id)


def make_smem_init_packets(device_id, cluster_id, size) :

	filler_data = get_fill_pattern(size, 0xDB500000)

	return h1ksim.make_cluster_mem_write_packets(
		h1ksim.Uint32Vector(filler_data),
		h1ksim.CLUSTER_SMEM_START,
		cluster_id,
		device_id)


def make_register_init_packets(device_id, cluster_id, tdsp_id) :

	regfile_size = h1ksim.TDSP_ADDR_MBOX_START

	filler_data = get_fill_pattern(regfile_size)

	return h1ksim.make_tdsp_write_packets(
		h1ksim.Uint32Vector(filler_data),
		0,
		tdsp_id,
		cluster_id,
		device_id,
		0)			# Event flags


def make_read_packets(addr_start, n_values, tdsp_id, cluster_id, device_id) :

	return h1ksim.make_tdsp_read_packets(
			addr_start,
			tdsp_id,
			cluster_id,
			device_id,
			n_values,
			RETURN_DEVICE_ID)


def make_read_packets(addr_start, n_values, cluster_id, device_id) :

	packets = []

	for tdsp_id in range(h1ksim.TDSPS_PER_CLUSTER) :
		packets.extend(make_read_packets(addr_start, n_values, tdsp_id, cluster_id, device_id))

	return packets


def tdsp_event_flags_to_string(integer):
	string = ''

	if integer & 0x80000000:
		string += 'OR: '
	integer &= 0x7FFFFFFF

	while integer:
		mask = integer & (~integer + 1)
		string += h1ksim.TdspEventFlagRegisterMasks_to_string(mask)[13:] + FLAG_TEXT_CONCAT_STRING
		integer ^= mask

	if len(string) > 0:
		return string[:-(len(FLAG_TEXT_CONCAT_STRING))]
	return string


def cluster_event_flags_to_string(integer):
	string = ''

	while integer:
		mask = integer & (~integer + 1)
		string += h1ksim.ClusterEventRegisterWriteMaksks_to_string(mask)[9:] + FLAG_TEXT_CONCAT_STRING
		integer ^= mask

	if len(string) > 0:
		return string[:-(len(FLAG_TEXT_CONCAT_STRING))]
	return string


def tdsp_status_reg_bits_to_string(integer):
	string = ''

	while integer:
		mask = integer & (~integer + 1)
		string += h1ksim.TdspStatusRegisterMasks_to_string(mask)[13:] + FLAG_TEXT_CONCAT_STRING
		integer ^= mask

	if len(string) > 0:
		return string[:-(len(FLAG_TEXT_CONCAT_STRING))]
	return string


def tdsp_cmd_register_values_to_string(integer):

	return {
		1:	'IDLE',
		2:	'RUN',
		4:	'STEP',
		5:	'STEP_IDLE',
		6:	'STEP_RUN',
		}.get(integer, '')


def tdsp_id_register_to_string(integer):

	device_id = (integer & 0xFFFFF00) >> 8
	cluster_id = (integer & 0xF8) >> 3
	tdsp_id = (integer & 0x7)

	return "Dev %d, Cls %d, tDSP %d" % (device_id, cluster_id, tdsp_id)


def feeder_control_reg_to_string(integer):

	value = integer & 0x3

	if value == 1:
		string = 'RUN'
	elif value == 3:
		string = 'PAUSED'
	elif value == 0:
		string = 'DISABLED'

	return string


PACKET_EVENTS = {
	1:	'EVFPKT0',
	2:	'EVFPKT1',
	4:	'EVFPKT2',
	8:	'EVFPKT3',
}


def packet_events_to_string(integer):
	string = ''

	while integer:
		mask = integer & (~integer + 1)
		string += PACKET_EVENTS[mask] + FLAG_TEXT_CONCAT_STRING
		integer ^= mask

	if len(string) > 0:
		return string[:-(len(FLAG_TEXT_CONCAT_STRING))]
	return string


def decode_flit_lo(integer):

	cluster_id = (integer & h1ksim.FLT_PHY_MASK_CLS) >> h1ksim.FLT_PHY_SHIFT_CLS
	tdsp_id = (integer & h1ksim.FLT_PHY_MASK_TDSP) >> h1ksim.FLT_PHY_SHIFT_TDSP
	flags = (integer & h1ksim.FLT_PHY_MASK_EVFM) >> h1ksim.FLT_PHY_SHIFT_EVFM
	address = (integer & h1ksim.FLT_PHY_MASK_ADDR3) >> h1ksim.FLT_PHY_SHIFT_ADDR3

	return "C:%d T:%d %s Addr:0x%05X" % (cluster_id, tdsp_id, packet_events_to_string(flags), address)


MEM_UNITS = {
	0:	"Invalid",
	h1ksim.FLT_MEM_UNIT_DEVICE:		"Dev",
	h1ksim.FLT_MEM_UNIT_CLUSTER:	"Cls",
	h1ksim.FLT_MEM_UNIT_TDSP:		"tDSP"
}


def decode_flit_hi(integer):

	size = (integer & h1ksim.FLT_MASK_SIZ_HW) >> h1ksim.FLT_SHIFT_SIZ_HW
	opcode = (integer & h1ksim.FLT_MASK_POP_HW) >> h1ksim.FLT_SHIFT_POP_HW
	m = (integer & h1ksim.FLT_MASK_M_HW) >> h1ksim.FLT_SHIFT_M_HW
	v = (integer & h1ksim.FLT_MASK_V_HW) >> h1ksim.FLT_SHIFT_V_HW
	device_id = (integer & h1ksim.FLT_PHY_MASK_DEV_HW) >> h1ksim.FLT_PHY_SHIFT_DEV_HW

	return "Size:%d Op:%s M:%s V:%s D:%d" % (size, h1ksim.FlitPacketOpcode_to_string(opcode)[8:], MEM_UNITS[m], "Phys" if v == 0 else "Virt", device_id)
