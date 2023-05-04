from threading import Thread
# Module used for communication
import socket
import json


class trafficlights(Thread):
    def __init__(self):
        """listener class. 
        
        Class used for running port listener algorithm 
        """
        super(trafficlights, self).__init__()
        
        # Semaphore states
        self.s1_state=0 
        self.s2_state=0 
        self.s3_state=0 
        self.s4_state=0 

        self._init_socket()

        #: Flag indincating thread state
        self.__running = True

    def _init_socket(self):
        # Communication parameters, create and bind socket
        self.PORT = 50007
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #(internet, UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.bind(('', self.PORT))
        self.sock.settimeout(1)

    def run(self):
        while self.__running:
            try:
                data, addr = self.sock.recvfrom(4096) 
                dat = data.decode('utf-8')
                dat = json.loads(dat)
                ID = int(dat['id'])
                state = int(dat['state'])
                if (ID == 1):
                    self.s1_state=state
                elif (ID == 2):
                    self.s2_state=state
                elif (ID == 3):
                    self.s3_state=state
                elif (ID == 4):
                    self.s4_state=state

            except Exception as e:
                print("Receiving data failed with error: " + str(e))

    
    def stop(self):
        """ Method for stopping listener process.
        """
        self.__running = False
