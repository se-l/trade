/ handle websocket messages
.z.ws: {value x}

/ handle closure of websockets
.z.wc: {delete from `subs where handle = x}

/ function to generate data
gendata: {
    / convert to a list if x is atom (to handle random generating function '?'
    if [0 > type x; x: 1#x];
    /jsonify the table
    .j.j flip `time`sym`val ! (.z.T + `minute$ asc neg[n]?100;n? x; (n: 1 + rand 10)?10.0)
    }

ClientResponse: {
 / process the input x
 response: gendata x;

 / send the response
 neg[.z.w] response;
 }

/ table that stores the subscribers data
subs: flip `handle`func`params! "is*" $\: ();

/ handles subscription
sub: {`subs upsert (.z.w; x; y)}

/ publish the data to subscribers
pub: {
    row: subs x;
    neg[row `handle] .j.j value[row `func] row `params
    }

/ timer function
.z.ts: {pub each til count subs}