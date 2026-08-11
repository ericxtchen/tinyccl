// ring_allreduce = reduce scatter + allgather
// do n - 1 steps of reduce scatter
// we receive some chunk -> run our reduce function on it -> send it to neighbor
// what if we had a thread that is in a worker loop waiting to receive data? how
// does this fit with our current recv method signature? have to open a socket
// connection on BASE_PORT + rank_ and listen from there once we get all data
// from recv, run the reduce function and then send do we even need multiple
// threads? we need to make sure there is no deadlock
