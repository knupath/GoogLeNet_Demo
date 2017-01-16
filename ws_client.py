from autobahn.twisted.websocket import WebSocketClientProtocol, \
    WebSocketClientFactory
from PIL import Image
import numpy as np
import base64
import json
import time
from os import listdir
from os.path import isfile, join

#fileName = 'cat_640x360.jpg'
#fileName = 'cat_256x256.jpg'
output_file = 'output_results.csv'
directory_name = 'pictures'


class MyClientProtocol(WebSocketClientProtocol):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.getImageList()
        fileExists = isfile(output_file)
        # Open output_file
        self.file_obj = open(output_file, "ab+")
        if not fileExists:
            self.file_obj.write("filename, #1 Name, #1 %, #2 Name, #2 %, #3 Name, #3 %\n")

    def getImageList(self):
        self.image_list = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]

    def sendImage(self):
        self.processImage()
        self.sendMessage(self.messageStr, isBinary=False)
        elaspsed_time = time.time() - self.t0
        self.frame_count += 1
        rate = self.frame_count / float(elaspsed_time)
        print('Frame Number: %d, Rate: %f'%(self.frame_count, rate ))

    def processImage(self):
        self.frame_count = 0
        self.fileName = self.image_list.pop()
        img =  np.asarray(Image.open(join(directory_name, self.fileName)), dtype='uint8')
        print('Image Loaded: ' + self.fileName)
        print img.shape
        print img.max()

          # shape should be (Nrows, Ncols, Nchannels)
        imgBytes = bytearray(img.flatten())
        imgBytes64 = base64.b64encode(imgBytes)
        message = {'image_ID': '0',
                   'width': img.shape[1],
                   'height':img.shape[0],
                   'data': imgBytes64}
        self.messageStr = json.dumps(message)

    def onConnect(self, response):
        print("Server connected: {0}".format(response.peer))

    def onOpen(self):
        print("WebSocket connection open.")

        self.t0 = time.time()
        self.sendImage()


    def onMessage(self, payload, isBinary):
        if isBinary:
            print("Binary message received: {0} bytes".format(len(payload)))
        else:
            # Write processed image's fileName
            self.file_obj.write(self.fileName)
            msg = json.loads(payload.decode('utf8'))
            msg = { key.encode('utf-8'):value for key,value in msg.items() }
            msg = msg['matches']
            print "\nTop 3 matches:"
            for match in msg:
                match = { key.encode('utf-8'):value for key,value in match.items() }
                print "%s: %d%%" % (match['name'],match['value']*100)
                # Write top 3 matches to output file
                self.file_obj.write(", %s, %d%%" % (match['name'],match['value']*100))
            self.file_obj.write("\n")
            self.file_obj.flush()
            #print("Text message received: {0}".format(payload.decode('utf8')))
            #self.sendImage()
            print ""
            if not self.image_list:
                print "List is Empty"
                self.sendClose()
            else:
                self.sendImage()

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))
        self.file_obj.close()
        self.factory.reactor.stop()


if __name__ == '__main__':

    import sys

    from twisted.python import log
    from twisted.internet import reactor

    log.startLogging(sys.stdout)
    # if len(sys.argv) == 2:
    #     fileName = sys.argv[1]

    factory = WebSocketClientFactory()  # (u"ws://127.0.0.1:4000", debug=False)

    factory.protocol = MyClientProtocol

    reactor.connectTCP("127.0.0.1", 9000, factory)
    reactor.run()
