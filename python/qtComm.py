from jupyter_client import find_connection_file, BlockingKernelClient
class qtComm():
	def __init__(self, command=None):
		self.cf = find_connection_file()
		print self.cf
		self.km = BlockingKernelClient(connection_file=self.cf);
		self.km.load_connection_file()
		if command is not None:
			self.km.execute(command)
	
	def send(self, command):
		return self.km.execute(command)

if __name__=='__main__':
	import sys
	comm=qtComm('TEST_STRING="344"; a=23;')
	if len(sys.argv)>1:
		print "Sending arg:", sys.argv[1]
		print comm.send(sys.argv[1])
	else:
		print comm.send('plot([1,4,7]);show()')