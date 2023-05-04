import sys
sys.path.insert(0,'.')

import socket
import json
import time

class Streamer:
	
	def __init__(self,server_data, streamPipe):
		"""Streamer aims to send all message to the server. 
		"""
		
		self.__server_data = server_data 
		self.socket_pos = None
		self.__streamP_pipe = streamPipe
		
		self.__running = True

	def stop(self):
		self.__running = False
		try :
			self.__server_data.socket.close()
		except: pass
		
	def stream(self):
		""" 
		After the subscription on the server, it's publishing the messages on the 
		previously initialed socket.
		"""
		while self.__running:
			if self.__server_data.socket != None: 
				try:
					msg = self.__streamP_pipe.recv()
					data = {'OBS': msg['obstacle_id'], 'x': msg['x'], "y": msg['y']}
					msg = json.dumps((data))
					try:
						self.__server_data.socket.sendall(msg.encode('utf-8'))
					except:
						self.__server_data.socket.sendall(msg)
					time.sleep(0.25)
					self.sent = True
				except Exception as e:
					self.__server_data.socket.close()
					self.__server_data.is_new_server = False
					self.__server_data.socket = None
					print("Sending data to server " + str(self.__server_data.serverip) + " failed with error: " + str(e))
					self.__server_data.serverip = None
				finally: 
					pass
			
		else:
			self.__server_data.is_new_server = False
			self.__server_data.socket = None
			self.__server_data.serverip = None
