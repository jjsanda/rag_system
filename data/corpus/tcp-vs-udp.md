# TCP vs. UDP

TCP and UDP are the two main transport-layer protocols of the Internet protocol suite. Both carry data between applications on top of the Internet Protocol (IP), but they differ in how much reliability and structure they add.

TCP, the Transmission Control Protocol, is connection-oriented. Before any data flows, the two endpoints set up a connection with a three-way handshake: one side sends a SYN, the other replies with a SYN-ACK, and the first answers with an ACK. From then on TCP guarantees that data arrives reliably and in order, retransmitting anything lost and acknowledging what it receives, and it adds flow control and congestion control so it does not overwhelm the receiver or the network.

UDP, the User Datagram Protocol, is connectionless. It sends independent packets, called datagrams, without setting up a connection and without promising that they arrive, arrive in order, or arrive only once. Giving up those guarantees is exactly what gives UDP its lower overhead and latency.

That trade-off decides where each one fits:

- TCP, where accuracy matters: web pages, email, and file transfers.
- UDP, where speed matters more than perfect delivery: live audio and video, online games, voice over IP, and DNS lookups.
