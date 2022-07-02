xc:{[m;f;x] /m- HTTP method,f - function name (sym), x - arguments
  /* execute given function with arguments, error trap & return result as JSON */
  if[not f in key .api.funcs;:.j.j "Invalid function"];                             //error on invalid functions
  if[not m in .api.funcs[f;`methods];:.j.j "Invalid method for this function"];     //error on invalid method
  if[count a:.api.funcs[f;`required] except key x;:.j.j "Missing required param(s): "," "sv string a]; //error on missing required params
  p:value[value f][1];                                                              //function params
  x:.Q.def[.api.funcs[f;`defaults]]x;                                               //default values/types for args
  :.[{.j.j x . y};(value f;value p#x);{.j.j enlist[`error]!enlist x}];              //error trap, build JSON for fail
 }

//prs:.req.ty[`json`form]!(.j.k;.url.dec)                                             //parsing functions based on Content-Type
getf:{`$first "?"vs first " "vs x 0}                                                //function name from raw request
getargstr:{last "?"vs x 0}
spltp:{0 1_'(0,first ss[x 0;" "])cut x 0}                                           //split POST body from params
prms:{.url.dec last "?"vs x 0}


ret:$[.z.K>=3.7;{.h.hy[1b;x;-35!(6;y)]};.h.hy];

\l sp.q
example:{select distinct p,s.city from sp}

.z.ph:{[x] /x - (request;headers)
  /* HTTP GET handler */
  //  :ret[`json] xc[`GET;getf x;prms x];                                               //run function & return as JSON
  // parse requested table from request and load from disk. the query and drop from mem again. for now, nothing's splayed...
  :ret[`json] .j.j string value .h.uh getargstr x;
 }

.z.pp:{[x] /x - (request;headers)
  /* HTTP POST handler */
  // CREATE: mode, table name, schema, data
  // UPSERT: mode, table name, data
  body:x 0;
//  body:spltp x 1;                                                                       //split POST body from params
  body:.h.xt[`json;("{\"\":\"\"}";body)] 1;
  // header:lower[key x 1]!value x 1;                                                  //lower case keys
  // show header;
  // a:prs[x[1]`$"content-type"]b[1];                                                  //parse body depending on Content-Type
  // if[99h<>type header;header:()];                                                   //if body doesn't parse to dict, ignore
  // header:@[header;where 10<>type each header;string];                               //string non-strings for .Q.def
//  show body;
  // loading table
  string value body[`tbl];
  result: string value body[`qsql];
//  show result;
  :ret[`json] .j.j result;                                                                //run function & return as JSON
 }

//.z.pp:{[x] /x - (request;headers)
//  /* HTTP POST handler */
//  b:spltp x;                                                                        //split POST body from params
//  x[1]:lower[key x 1]!value x 1;                                                    //lower case keys
//  a:prs[x[1]`$"content-type"]b[1];                                                  //parse body depending on Content-Type
//  if[99h<>type a;a:()];                                                             //if body doesn't parse to dict, ignore
//  a:@[a;where 10<>type each a;string];                                              //string non-strings for .Q.def
//  :ret[`json] xc[`POST;getf x;a,prms b];                                            //run function & return as JSON
// }

show `$"API started on 5052"
\p 5042 / server