root: "/repos/trade/data/kdb"
path: {[fn] hsym `$ "/" sv (root;fn)}

mktrades:{[tickers; sz]
  dt:2015.01.01+sz?31;
  tm:sz?24:00:00.000;
  sym:sz?tickers;
  qty:10*1+sz?1000;
  px:90.0+(sz?2001)%100;
  t:([] dt; tm; sym; qty; px);
  t:`dt`tm xasc t;
  t:update px:6*px from t where sym=`goog;
  t:update px:2*px from t where sym=`ibm;
  t}
trades:mktrades[`aapl`goog`ibm; 1000000]

p: path["trades"]

p set trades