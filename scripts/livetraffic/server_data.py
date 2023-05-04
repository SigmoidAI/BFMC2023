import sys
sys.path.insert(0,'.')

"""ServerData class contains all parameter of server. It need to connect to the server.
The parameters is updated by other class, like ServerListener and SubscribeToServer
"""
class ServerData:

	def __init__(self, beacon_port, server_IP = None):
		#: ip address of server 
		self.__server_ip = server_IP 
		#: flag to mark, that the server is new. It becomes false, when the client subscribed on the server.
		self.is_new_server = False
		#: port, where the beacon server send broadcast messages
		self.__beacon_port = beacon_port
		#: port, where the server listen the car clients
		self.carSubscriptionPort = None
		#: connection, which used to communicate with the server
		self.socket = None

	
	@property
	def beacon_port(self):
		return self.__beacon_port

	@property
	def serverip(self):
		return self.__server_ip

	@serverip.setter
	def serverip(self, server_ip):
		if self.__server_ip != server_ip:
			self.__server_ip = server_ip
			self.is_new_server = False
