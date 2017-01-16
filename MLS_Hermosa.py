import struct
import math
import numpy as np
import logging
import sys
import time
import os
import sqlite3 as lite
import re

import base64
import json
from autobahn.twisted.websocket import WebSocketServerProtocol
from autobahn.twisted.websocket import WebSocketServerFactory

from twisted.python import log
from twisted.internet import reactor

import scipy.misc

from fpga_api.dut_remote import RemoteDut
from fpga_api.Header import Header1K
import fpga_api.pyh1k


logging.basicConfig(level=logging.DEBUG,format='[%(levelname)s] (%(threadName)-10s) %(message)s' )

# All the tasks are put into the queue as they are received
# The image classifier thread processes the tasks in order and sends back results to the GUI
#taskQ = Queue()


class DB(object):
    """
    All functions related to the database
    """
    def __init__(self, _category_obj, _db_filename ='GoogLeNet_DB.sqlite'):

        # Check if file exists
        db_file_exists = os.path.exists(_db_filename)

        # Creates and configures DB objects
        self.db_connect = lite.connect(_db_filename)

        # Enables Foreign Keys to implement table constraints
        self.db_connect.execute('pragma foreign_keys=ON')
        self.db_cursor = self.db_connect.cursor()

        # Initializations
        self.category_obj = _category_obj
        self.category_count = _category_obj.get_category_count()
        self.last_src = ''
        self.last_time = 0
        self.image_json = {}
        self.include_file_record = False

        # Construct Table Headers
        self.file_table_header = [('fileId', 'INTEGER PRIMARY KEY'), ('width', 'INTEGER'), ('height', 'INTEGER'), ('src', 'TEXT')]
        self.frame_table_header = [('fileId', 'INTEGER'), ('image_ID', 'TEXT'), ('time', 'REAL')]
        self.frame_table_prefix_count = len(self.frame_table_header)

        for x in range(self.category_count):
            self.frame_table_header.append((self.category_obj.get_header(x), 'TEXT'))

        # Construct Insert Record Commands
        self.insert_file_info_string = "INSERT INTO file_table VALUES(" + \
            ", ".join("?" for i in range(len(self.file_table_header))) + \
            ")"
        self.insert_frame_info_string = "INSERT INTO frame_table VALUES(" + \
                ", ".join("?" for i in range(len(self.frame_table_header))) + \
                ")"

        # Ensure schema exists, if not initialize
        if not db_file_exists:
            self.init_schema()

        # Obtain last fileId
        self.db_cursor.execute("SELECT fileId FROM file_table")
        rows = self.db_cursor.fetchall()
        rows = [element[0] for element in rows] # changes list of tuples returned by sqlite to a list
        if len(rows) == 0:
            self.last_file_id = 0
        else:
            self.last_file_id = max(rows)

    def update_image_json(self, _dict):
        """
        Retrieves image_json dictionary from plug-in
        """
        self.image_json = _dict

    def init_schema(self):
        """
        Initializes and creates database schema
        """
        create_file_table = "CREATE TABLE file_table(" + ", ".join('%s %s' % column for column in self.file_table_header) + ")"

        # FOREIGN KEY(fileId) REFERENCES file_table(fileId) at the end of the query implements a table constraint
        create_frame_table = "CREATE TABLE frame_table(" + ", ".join('%s %s' % column for column in self.frame_table_header) + ", FOREIGN KEY(fileId) REFERENCES file_table(fileId) )"

        self.db_cursor.execute(create_file_table)
        self.db_cursor.execute(create_frame_table)

    def update_file_record(self):
        """
        Create a new record if its a new src or if the video has been re-winded/restarted
        """
        if self.last_src != self.image_json['src'] or self.image_json['time'] < self.last_time:
            self.include_file_record = True
            self.last_src = self.image_json['src']
        self.last_time = self.image_json['time']

    def insert_new_record(self, probability_results):
        """
        Inserts new records into the db
        """
        self.update_file_record()

        # New File Record
        if self.include_file_record:
            self.include_file_record = False
            self.last_file_id += 1
            db_record = [self.last_file_id]
            for column in self.file_table_header[1:]:
                db_record.append(self.image_json[column[0]])
            self.db_cursor.execute(self.insert_file_info_string, db_record)

        # Update Frame Record
        db_record = [self.last_file_id]
        for column in self.frame_table_header[1:self.frame_table_prefix_count]:
            db_record.append(self.image_json[column[0]])
        db_record.extend(probability_results)
        self.db_cursor.execute(self.insert_frame_info_string, db_record)

        # Commit after each record update for now since MLS_Hermosa.py doesn't have an elegant way of shutting down
        self.commit()

    def commit(self):
        """
        Commit changes to db
        """
        self.db_connect.commit()

    def __del__(self):
        """
        To ensure changes are committed when class is destructed
        """
        self.commit()


class Categories(object):
    """
    All functions relating to accessing and retrieving MLS_Hermosa.py categories
    """
    def __init__(self):
        """
        Reads and parses category file into a list
        """
        category_src = 'synset_words.txt'
        category_input = np.loadtxt(category_src, str, delimiter='\t')
        self.categories = []
        self.categories_truncated = []
        max_category_length = 15
        pattern = re.compile('\W+')
        for s in category_input:
            label = s.split(' ', 1)[1].split(',')[0].replace(" ", "_").replace("-", "_").lower()
            label = pattern.sub('', label)  # Removes all non-alphanumeric + _
            while label in self.categories:
                label += "_"
            self.categories.append(label)
            self.categories_truncated.append(label[:max_category_length])

    def get_category_count(self):
        return len(self.categories)

    def get_header(self, index):
        return self.categories[index]

    def get_truncated_header(self, index):
        return self.categories_truncated[index]


# ********************************************************************
# Server implementation class
# - Inherit from WebSocketServerProtocol
# - Implement onMessage() function
# - call self.sendMessage() to send a message to the client
class MyServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.frame_count = 0
    def onMessage(self, payload, isBinary):
        if isBinary:
            # Note: We are not expecting binary messages for this
            # server
            str = "ws_server - Binary message received: {0} bytes".format(len(payload))
            logging.debug(str)
            # Send back a text reply
            self.sendMessage(str, False)
        else:
            logging.debug("Text message received: {0} bytes".format(len(payload)))
            # Text message should have the following JSON format:
            # {
            #     "image_ID":"0x0003000100020005",
            #     "width":640,
            #     "height":360,
            #     "data":"dfslkfdsklsfsl... (lots more pixel data here)"
            # }
            # The fields are:
            # image_ID: An identifier for the video frame, as provided
            #           by the browser add-on. This sting value should
            #           be returned to the browser over the WebSocket
            #           unmodified. (Javascript cannot natively represent
            #           64-bit integers, so the intent is for this value
            #           to be presented as a Javascript string.)
            # width:    Video frame width in pixels
            # height:   Video frame height in pixels
            # data:     Base64 encoded video frame pixel data in RGBA
            #           order, 1 byte per component, in row-major order
            #           (left to right), from the top of the image to
            #           the bottom.
            # src:      String with the YouTube URL from the browser.
            # time:     Floating point value that has the number of seconds
            #           from the start of the video

            logging.debug('Frame num : %d'% self.frame_count)

            # Retrieve and parse info from input stream
            image_json         = json.loads(payload.decode('utf8'))
            if not 'data' in image_json:
                logging.error('No image found!')
                return
            image_json['data'] = base64.b64decode(image_json['data'])
            image_json['data'] = bytearray(image_json['data'])
            if not 'width' in image_json:
                image_json['width'] = 0
            if not 'height' in image_json:
                image_json['height'] = 0
            if not 'image_ID' in image_json:
                image_json['image_ID'] = 'N/A'
            if not 'src' in image_json:
                image_json['src'] = 'N/A'
            # If time is not provided, use frame count
            if not 'time' in image_json:
                image_json['time'] = self.frame_count

            logging.debug(
                    'Image ID = {0}, width = {1}, height = {2}, data.length = {3}, src = {4}, time = {5}'.format(
                        image_json['image_ID'], image_json['width'], image_json['height'], len(image_json['data']), image_json['src'], image_json['time']))

            # dont process the funny cases that might come up
            if image_json['height'] == 0 or image_json['width'] == 0:
                logging.error('Frame with size zero received')
                return

            # Channels: RGBA
            if len(image_json['data'])== image_json['width']*image_json['height']*4:
                imageFull = np.reshape(np.array(image_json['data']), (image_json['height'], image_json['width'], 4))
                # throw away the alpha channel
                image = imageFull[:,:,0:3].astype('float32')
            # or else channels: RGB
            elif len(image_json['data'])== image_json['width']*image_json['height']*3:
                imageFull = np.reshape(np.array(image_json['data']), (image_json['height'], image_json['width'], 3))
                image = imageFull.astype('float32')
            else:
                return # dont queue the frame if it is strange

            # The image should have shape (NumRows, NumCols, NumChannels)
            logging.debug('Putting image %s in queue' % image_json['image_ID'])
            # put the image data and the ref to Websocket object into the task queue
            imageTask = (image, image_json['image_ID'], self, self.frame_count)
            # taskQ.put(imageTask, block=False)


            #### HACK... should not use db_obj #####
            db_obj.update_image_json(image_json)

            mls.processImages(imageTask)

            self.frame_count = self.frame_count + 1


    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {}".format(reason))
        #reactor.stop()

    def onConnect(self, request):
        print("Client connecting: {}".format(request.peer))


class MLS:
    def __init__(self, dutHostName_mls, category_obj_mls, db_obj_mls):
        self.dutHostName = dutHostName_mls
        self.category_obj = category_obj_mls
        self._db_obj = db_obj_mls
        self.running = True
        self.dut = None
        self.Nprobs = 1000
        self.initialize(self.dutHostName)

    def tileImage(self, raw_img):
        """
        split a 256x256x3 image up into 32 tiles. Each tile will have size of (64, 32, 3).
        The image will be divided into 8 slices horizontally and 4 slices vertically.
        Also subtract the channel mean.
        """
        assert(raw_img.shape ==(256,256,3))

        img = raw_img    - self.channel_mean
        Nc = 3
        Nx = 32
        Ny = 64
        flat_tiles = []
        for iy in range(4):
            for ix in range(8):
                tile = img[iy*Ny:(iy+1)*Ny, ix*Nx:(ix+1)*Nx]
                # flatten the 3d images so that depth index varies fastest, then row index, then column index
                flat_tiles.append(tile.transpose(1,0,2).flatten())
        return flat_tiles

    def sendDrainConfigurationFlit(self, dev_id):
        """
        Send drain configuration flit and wait for response
        """
        # m = 1 (device controller) , opcode = 1 (write)
        # addr = 0x410 + 1 for drain timer 1
        header = Header1K(size=7, opcode=1, m=1, device_id=dev_id, addr= 0x411)
        # Response header
        response = Header1K(size=2, opcode=1, m=1, device_id=99)
        transaction_ID      = 0x0000BEAF
        cluster_drain_mask  = 0xFFFFFFFF
        misc_drain_mask     = 0xFFFFFFFF
        timer_value         = 120000
        event_mask          = 0x00000002
        drain_flit = [header.encode(), 0]
        # pack the longwords together into quadwords
        drain_flit.append(transaction_ID    | (cluster_drain_mask<<32)   )
        drain_flit.append(misc_drain_mask   | (timer_value<<32)          )
        drain_flit.append(event_mask)

        # send the drain configuration flit to the cluster drain controller
        self.dut.write([drain_flit])
        # wait for the response

        # response = self.dut.listen(time_ms = 30000, n_flits = 1)
        # if response and len(response) >= 1 and len(response[0]) >= 4:
        #     resp_transaction_ID = response[0][2]
        #     resp_error_code = response[0][3]
        #     logging.debug('Drain response:')
        #     logging.debug('Transaction ID: ' + hex(resp_transaction_ID) )
        #     logging.debug('Error code: ' + str(resp_error_code))
        # else:
        #     logging.error('Drain response error')


    def sendTilesToHermosa(self, flat_tiles, dev_id):
        """
        flat_tiles is a list of 32 image tiles.  Each image tile is a 1d np array with 64*32*3 floats.
        The image tiles are put into packets and distributed to the 32 clusters of Hermosa with dev_id = 0 or 1.
        After sending out the image tiles a drain configuration flit is sent.
        """

        initial_offset = 630     # initial offset to account padding on the left
        nfloats = 6144           # number of longwords per image tile
        start_addr = 0x0008E1B5     # sMEM address to put the image tile at
        # start_addr  = 0x00080000    # cluster 0,4,16,20
        maxPacketSize = 16          # max number of 64-bit words in payload
        numFloatsPerPacket = 32     # 32 32-bit floats per packet
        sp = struct.Struct('<16Q')  # each packet will have up to 16 quadwords
        numFullPackets = 192        # = nfloats/numFloatsPerPacket = 64*32*3/32 = 192
        packetsPerColumn = 6        # = 64*3/32.
        # After every 6 packets we need to advance the address forward to account for padding
        padding_advance = 15        #  additional address advance at the end of each column

        assert(len(flat_tiles[0]) == nfloats)
        logging.debug('Sending %d bytes to K%d'%(nfloats*4*32, dev_id))

        # opcode=1 (write), m = 2 (cluster memory)
        h = Header1K(size=numFloatsPerPacket, opcode=1, m=2, device_id=dev_id,
                             cluster_id=0, addr=start_addr, v=0 )
        #loop over tiles/clusters
        for ic in range(32):
            n = 0
            curr_addr = start_addr + initial_offset
            curr_tile = flat_tiles[ic]
            # change cluster index in header
            h.cluster_id = ic
            packets = []
            # split up the data into packets of up to 16 64-bit words
            # each element of the packet list is 64 bits
            for i in range(numFullPackets):
                h.addr = curr_addr
                packetInts = sp.unpack_from(curr_tile[n:n+numFloatsPerPacket])
                packet = [h.encode()]
                packet.extend(packetInts)
                packets.append(packet)
                curr_addr += numFloatsPerPacket
                # additional advance at the end of each columnn to leave space for padding
                if (i+1)%packetsPerColumn==0:
                    curr_addr += padding_advance
                n += numFloatsPerPacket

            # write all the packets for a cluster at once
            self.dut.write(packets)

        # time.sleep(10)
        self.sendDrainConfigurationFlit(dev_id)


    def waitForResponse(self):
        """
        Wait for a reponse of Nprobs = 1000 floats from the Hermosa.
        Returns a numpy array of Nprobs = 1000 floats, and a cycle count for timing
        """
        cycleCount = 0
        probs = np.zeros(self.Nprobs)
        Nflits = 35  # The 1000 probs will be recieved in 34 flits, plus one flit for timing info
        response = self.dut.listen(time_ms = 20000, n_flits = Nflits)
        # for resp in response:
        #     print( hex(resp[0]) + ': ' + str(resp[1:]) )
        if len(response) == Nflits:
            for flit in response:                   # loop over flits, not neccasarly in order so get the flit index
                flit_index = flit[0]
                payload_size = len(flit) - 2

                if flit_index < self.Nprobs:        # probability flit
                    for i in range(payload_size):   # loop over longwords in the payload, each represents a float
                        # reinterpret the 32-bit int as a 32-bit float.
                        probs[flit_index + i]  = struct.unpack('f', struct.pack('I', flit[i + 2]))[0]
                else:                               # timing flit
                    cycleCount = flit[2]            # 32-bit int cycle count

            logging.debug('Cycle count: ' + str(cycleCount))
            logging.debug('Total probability: ' + str(probs.sum()))
        else:
            # send back a fake response if no response is received before timeout
            logging.error('Waiting for Response timed out')
            probs[0] = 1.0

        doSoftMax = False
        if doSoftMax:
            probs_sub = probs - probs.max()
            expProbs = np.exp(probs_sub)
            probs = expProbs/expProbs.sum()
            logging.debug('After softmax total probability: ' + str(probs.sum() ) )
        return (probs, cycleCount)

    def classifyFrame(self, input_image):
        """
        Split up in the input_image into sub-frames of size 256x256 (for now we only use 1 sub-frame).
        Calls tileImage to split up each sub-frame, then calls sendTilesToHermosa to send the tiles to Hermosa dev_id = 0 or 1.
        Then calls waitForResponse which receives the results back from Hermosa.
        This function returns a list of the top 3 matches in the format needed by the GUI.
        """
        dev_id = self.DEV_ID
        Nx = self.Nx
        height = input_image.shape[0]
        width = input_image.shape[1]
        y0 = max(0, (height-Nx)/2)
        x0 = max(0, (width -Nx)/2)
        cropped_center  = input_image[y0:y0+Nx,    x0:x0+Nx,  :]
        if cropped_center.shape[0] != Nx or cropped_center.shape[1] != Nx:
            cropped_center  = scipy.misc.imresize(cropped_center, (Nx, Nx, 3))

        tiles = self.tileImage(cropped_center)
        self.sendTilesToHermosa(tiles, dev_id)

        # for now we are using a single sub-frame
        (probs, cycleCount) = self.waitForResponse()

        # update db with returned results
        self._db_obj.insert_new_record(probs)

        # sort from highest to lowest probability
        sortedPredictions = np.argsort(-probs)
        matches = []
        # send back the top three
        for n in range(3):
             matches.append({'name'  :   self.category_obj.get_truncated_header(sortedPredictions[n]),
                             'value' :   float(probs[sortedPredictions[n]] ) })
        return (matches, cycleCount)


    def initialize(self, dutHostName):
        """
        Connect to the DUT board, and send in the GoogLeNet fhex file.
        Also load the list of 1000 categories
        """

        FLIT_FILE1 = 'cnn1.fhex'
        FLIT_FILE2 = 'cnn2.fhex'
        self.DEV_ID = 0
        self.dut = RemoteDut(host=dutHostName, devices=1, clusters=32)
        self.dut.reset(self.dutHostName)
        self.dut.localgo()

        # send in the fhex file
        logging.debug('Reading flit file...')
        flitStream1 = fpga_api.pyh1k.read_hexfile_as_64(FLIT_FILE1)
        flitStream2 = fpga_api.pyh1k.read_hexfile_as_64(FLIT_FILE2)
        flitStream = flitStream1 + flitStream2
        packets = self.create_packet_tuples(flitStream)
        logging.debug('Writing ' + str(len(packets)) + ' packets to ' + dutHostName)
        self.dut.write(packets)

        # request hacky write confirmation
        read = Header1K(
                size=2,
                opcode=0,
                m=3,
                cluster_id=31,
                tdsp_id=7
            )
        response = Header1K(
                    header=read,
                    size=0,
                    opcode=1,
                    m=1,
                    device_id=self.dut.device_id
                )
        logging.debug("Sending write confirmation packet...")
        self.dut.write([[read.encode(), response.encode()]])
        write_response = self.dut.listen(40 * 1000, n_flits=1)
        if not len(write_response):
            logging.warn("No write confirmation received")
        else:
            logging.debug("Received write confirmation")
        self.Nx = 256
        self.channel_mean = np.array([104., 117., 123.]).astype('float32')


    def processImages(self, task):
        """
        Wait for an image to appear on the queue. When it arrives, pull it off the queue and send it to the classifyFrame function.
        Then send back the response to the GUI.
        """

        logging.debug('Starting image processing thread')
        frameCount = 0
        #while self.running:
       # task = taskQ.get()  #  block until task is ready
        image = task[0]
        image_ID = task[1]
        webSocket = task[2]
        frameNum = task[3]
        #logging.debug('Queue size: %d' % taskQ.qsize())
        logging.debug('%d, Processing Frame Number: %d'% (frameCount, frameNum))
        (matches, cycleCount) = self.classifyFrame(image)
        # processing time in ms.  Is one cycle exaclty 1 ns?
        proc_time = cycleCount/1.0e6
        response = { 'image_ID' : image_ID, 'matches' : matches, 'proc_time': proc_time}
        respStr = json.dumps(response)
        logging.debug(respStr)
        webSocket.sendMessage(respStr, False)
        frameCount += 1
        #taskQ.task_done()


    def create_packet_tuples(self, stream) :
        """
        Turn stream of 64 bit ints into list of packet tuples of the form:
        [[header, payload1, ....], [header, ...], ...]
        Each element of the tuple is a 64 bit int
        This code was taken from fpga_api.flit_utils and changed slightly
        """
        packets = []
        i = 0
        while i < len(stream) :
            #size = h1ksim.FlitPacketHeader.extract_size(stream[i])
            size = stream[i]>>58
            #Get number of 64 bit ints this packet spans.
            size = ((size + 1) / 2) + 1
            packets.append(tuple(stream[i:i + size]))
            i += size

        return packets


    def startServer(self, portNum):
        """
        Start up the websocket server.
        """

        # start the image processing thread which will wait for images to enter the queue
        #self.imageProcessingThread =  Thread(target=self.processImages)
        #self.imageProcessingThread.setDaemon(True)
        #self.imageProcessingThread.start()
        log.startLogging(sys.stdout)

        factory = WebSocketServerFactory()
        factory.protocol = MyServerProtocol

        # disabling the utf8 validation causes a 40% decrease in the processing time per frame
        # does this cause any problems?
        factory.setProtocolOptions(utf8validateIncoming = False)

        reactor.listenTCP(portNum, factory)
        # start the websocket server
        reactor.run()


    def __del__(self):
        self.dut.reset(self.dutHostName)

if __name__== '__main__':
    if len(sys.argv) < 2:
        print('SDB Host Name required')
        print('Usage: python MLS_Hermosa.py <SDB_HOST_NAME>')
    else:
        portNum = 9000
        category_obj = Categories()
        db_obj = DB(category_obj)
        mls = MLS(sys.argv[1], category_obj, db_obj)
        mls.startServer(portNum)
