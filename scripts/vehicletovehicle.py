from threading import Thread
# Module used for communication
import socket
import json

##  listener class. 
#
#  Class used for running port listener algorithm 
class vehicletovehicle(Thread):
    
    def __init__(self):
        """listener class. 
        
        Class used for running port listener algorithm 
        """
        super(vehicletovehicle,self).__init__()

        # Values extracted from message
        self.ID = 0
        self.pos = complex(0,0)

        self._init_socket()

        # Flag indincating thread state
        self.__running = True

    def _init_socket(self):
        # Communication parameters, create and bind socket
        self.PORT = 50009
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #(internet, UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.bind(('', self.PORT))
        self.sock.settimeout(1)


    def run(self):
        while self.__running:
            try:
                data,_ = self.sock.recvfrom(4096)
                data = data.decode("utf-8") 
                data = json.loads(data)

                self.pos = complex(data['coor'][0], data['coor'][1])

                self.pos = complex(data['coor'])
            except Exception as e:
                print("Receiving data failed with error: " + str(e))

    ## Method for stopping listener process.
    #  @param self          The object pointer.
    def stop(self):
        self.__running = False
