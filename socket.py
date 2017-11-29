import socketserver
import socket
import numpy as np

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        while 1:
            self.data = self.request.recv(1024).strip()
            print("{} wrote:".format(self.client_address[0]))
            print(self.data)
            # just send back the same data, but upper-cased
##            self.request.sendall(self.data.upper())
            self.request.sendall(np.array([1,2,3,4]))
            if 'd' in str(self.data):
                break

if __name__ == "__main__":
    HOST, PORT = socket.gethostname(), 9999
    
    # Create the server, binding to localhost on port 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
