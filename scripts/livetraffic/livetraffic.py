from threading import Thread
from multiprocessing import Pipe
from .server_data import ServerData
from .server_listner import ServerListener
from .server_subscriber import ServerSubscriber
from .streamer import Streamer

import time
import random

class EnvironmentalHandler(Thread):
    
    def __init__(self, ID, beacon, serverpublickey, streamPipe, clientprivatekey):
        """ EnvironmentalHandler targets to connect on the server and to send messages, which incorporates 
        the coordinate of the encountered obstacles on the race track. It has two main state, the setup state and the streaming state. 
        In the setup state, it creates the connection with server. It's sending the messages to the server in the streaming
        state. 
        It's a thread, so can be run parallel with other threads. You can write the coordinates and the id of the encountered obstacle 
        and the script will send it.
        """
        super(EnvironmentalHandler, self).__init__()
        #: serverData object with server parameters
        self.__server_data = ServerData(beacon)
        #: discover the parameters of server
        self.__server_listener = ServerListener(self.__server_data)
        #: connect to the server
        self.__subscriber = ServerSubscriber(self.__server_data, ID, serverpublickey, clientprivatekey)
        #: receive and decode the messages from the server
        self.__streamer = Streamer(self.__server_data, streamPipe)
        
        self.__running = True

    def setup(self):
        """Actualize the server's data and create a new socket with it.
        """
        # Running while it has a valid connection with the server
        while(self.__server_data.socket == None and self.__running):
            # discover the parameters of server
            self.__server_listener.find()
            if self.__server_data.is_new_server and self.__running:
                # connect to the server 
                self.__subscriber.subscribe()
        
    def stream(self):
        """ Listening the coordination of robot
        """
        self.__streamer.stream()

    def run(self):
        while(self.__running):
            self.setup()
            self.stream()
    
    def stop(self):
        """Terminate the thread running.
        """
        self.__running = False
        self.__server_listener.stop()
        self.__streamer.stop()

