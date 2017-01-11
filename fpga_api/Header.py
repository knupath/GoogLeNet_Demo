

# Constants
H2000 = 0
H1000X = 1


#-----------------------------------------------------------------------------
def make_mask(msb, size):
	return (2**msb - 1) ^ (2**(msb-size) - 1)


##############################################################################
class Header(object):

	WORD_SIZE = 64
	
	standard_fields = ()
	format_index = "m"
	formats = {}

	#-------------------------------------------------------------------------
	def __init__(self, header = 0, **kwargs):
	
		if isinstance(header, Header):
		
			# Copy header
			for field_name in header._get_field_names():
				setattr(self, field_name, getattr(header, field_name))

		else:
	
			qword = header
			msb = self.WORD_SIZE
			
			# Set standard header fields
			for name, size in self.standard_fields:
				setattr(self, name, (qword & make_mask(msb, size)) >> (msb - size))
				msb -= size
			
			# If format is given, set it so we can pick the right header format
			if self.format_index in kwargs:
				setattr(self, self.format_index, kwargs[self.format_index])
				
			format = self._get_variable_fields()

			# Extract special format fields from header bytes
			for name, size in format:
				setattr(self, name, (qword & make_mask(msb, size)) >> (msb - size))
				msb -= size
				
		# Override any fields with keyword arguments
		for attr, value in kwargs.iteritems():
			setattr(self, attr, value)
				
				
	#-------------------------------------------------------------------------
	def _get_variable_fields(self):
	
		try:
			return self.formats[getattr(self, self.format_index)]
		except KeyError:
			raise RuntimeError(
				"Invalid %s field value: %d."
				% (self.format_index, getattr(self, self.format_index)))
				
				
	#-------------------------------------------------------------------------
	def _get_format(self):
		return self.standard_fields + self._get_variable_fields()


	#-------------------------------------------------------------------------
	def _get_field_names(self):
		return [field_name for field_name, _ in self._get_format()]
		
		
	#-------------------------------------------------------------------------
	def _get_field(self, field_name):
	
		try:
			return getattr(self, field_name)
		except AttributeError:
			return 0
		
		
	#-------------------------------------------------------------------------
	def encode(self):
	
		header = 0
		shift = 0
		
		# Start with last field first
		for field_name, width in reversed(self._get_format()):
		
			value = self._get_field(field_name)
			
			# Check value fits in field width
			trunc_value = value & (2**width - 1)
			if value != trunc_value:
				raise RuntimeError("Field value doesn't fit in field width.  "
							"Field: %s, Width: %d, Value: 0x%06X" % (field_name, width, value))
							
			header |= (value << shift)
			
			shift += width
			
		return header
		
		
	#-------------------------------------------------------------------------
	def encode32(self):
	
		h = self.encode()
		return (h & 0xFFFFFFFF), ((h >> 32) & 0xFFFFFFFF)
		
		
	#-------------------------------------------------------------------------
	def __str__(self):
	
		string = self.__class__.__name__ + "\n  "
		
		string += '\n  '.join(
					["%-12s: %d" % (field_name, self._get_field(field_name)) for field_name in self._get_field_names()]
				)
				
		string += "\n\n  %016X" % self.encode()
		string += "\n  %08X %08X" % self.encode32()
				
		return string

			
##############################################################################
class Header1K(Header):

	standard_fields = (("size", 6), ("opcode", 3), ("m", 2), ("v", 1), ("device_id", 20))
	
	formats = {
		0:	(("addr", 32),),
		1:	(("addr", 32),),
		2:	(("cluster_id", 5), ("addr", 27)),
		3: 	(("cluster_id", 5), ("tdsp_id", 3), ("evfm", 4), ("addr", 20)),
	}
	

##############################################################################
class Header2K(Header):

	standard_fields = (("size", 5), ("opcode", 3), ("m", 2), ("device_id", 19))
	
	formats = {
		1:	(("addr", 32),),
		2:	(("cluster_id", 5), ("addr", 27)),
		3: 	(("cluster_id", 5), ("tdsp_id", 3), ("evfm", 4), ("addr", 20)),
	}