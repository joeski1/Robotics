#!/usr/bin/env python3
# proxy a TCP socket through a server
# @author mbway

'''  Explanation

./socket-proxy.py input_port output_port

Two clients connect to one of the server ports and receive client sockets which
are at random ports. Any data sent from the input side is relayed to the output
side

--> :input_port   |                 |
                  | socket-proxy.py |
                  |                 | :output_port   <--

Now with client sockets established
--> |                 |
    | socket-proxy.py |
    |                 | -->



./socket-proxy.py input_port output_ip:output_port

Again a server port is bound for the input side, but now the output client socket
is established by connecting to a known location

--> :input_port   |                 |
                  | socket-proxy.py |
                  |                 | output client  --> output_ip:output_port

Now with client sockets established
--> |                 |
    | socket-proxy.py |
    |                 | -->
'''

import sys
import socket

input_host  = socket.gethostname()
output_host = socket.gethostname()

def set_hosts_by_interface(input_iface, output_iface):
    import netifaces as ni
    global input_host
    global output_host
    input_host  = ni.ifaddresses(input_iface)[ni.AF_INET][0]['addr']
    print('input host ip:  ' + input_host)
    output_host = ni.ifaddresses(output_iface)[ni.AF_INET][0]['addr']
    print('output host ip: ' + output_host)

def print_message():
    if input_host != output_host:
        if output_to_loc:
            print('input {}:{} redirected to output location {}:? -> {}:{}'
                .format(input_host, input_port, output_host, output_server, output_port))
        else:
            print('input {}:{} redirected to output {}:{}'
                .format(input_host, input_port, output_host, output_port))
    else:
        host = input_host # both the same
        if output_to_loc:
            print('input port {} redirected to output location {}:{} on host {}'
                .format(input_port, output_server, output_port, host))
        else:
            print('input port {} redirected to output port {} on host {}'
                .format(input_port, output_port, host))


if __name__ == '__main__':

    if len(sys.argv) < 2+1:
        print('not enough arguments: ./socket-proxy input_port output_location_or_port [input_interface, output_interface]')
        sys.exit(1)
    if len(sys.argv) == 4+1:
        print('input interface:  ' + sys.argv[3])
        print('output interface: ' + sys.argv[4])
        set_hosts_by_interface(sys.argv[3], sys.argv[4])

    input_port  = int(sys.argv[1])

    # whether to output to a location (ie server:port) or listen on a local port
    output_to_loc = ':' in sys.argv[2]
    if output_to_loc:
        [output_server, output_port] = sys.argv[2].split(':')
        output_port = int(output_port)
    else:
        output_port = int(sys.argv[2])

    print_message()


    # create server sockets for input, and output only if not outputting to
    # another location
    input_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    input_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    input_srv.bind((input_host, input_port))
    input_srv.listen(1)

    if not output_to_loc:
        output_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        output_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        output_srv.bind((output_host, output_port))
        output_srv.listen(1)

    input_cli  = None
    output_cli = None

    while True:
        try:
            if output_cli is None:
                if output_to_loc:
                    print('connecting to output location')
                    output_cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    output_cli.connect((output_server, output_port))
                else:
                    print('waiting for output')
                    (output_cli, _) = output_srv.accept()

            if input_cli is None:
                print('waiting for input')
                (input_cli, _) = input_srv.accept()

            if output_cli is not None and input_cli is not None:
                print('both parties connected')


            while True:

                data = input_cli.recv(4096)
                if len(data) == 0:
                    input_cli.close()
                    input_cli = None
                    break
                #print('relaying packet of ' + str(len(data)) + ' bytes')
                output_cli.sendall(data)

        except BrokenPipeError:
            output_cli.close()
            output_cli = None
            input_cli.close()
            input_cli = None

