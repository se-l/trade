# README #

Test model with generation with default parameters and data in repo.

### What is this repository for? ###

* Generate trading strategies
* Backtest strategies
* Evaluate strategy performance

### How do I get set up? ###

#### Windows tested - Linux later
- didnt check the access issues on how to clone it... checking later
```console
git clone git@bitbucket.org:SebastianLuen/trade2.git  
cd trade2  
pip install -r requirements.txt
```
the last statement is likely to have problems. need to run as required when packages are missing. see file comments.

hyperopt installation likely to fail from pypi. Therefore,  
pip install --upgrade git+git://github.com/hyperopt/hyperopt.git

Compile Cython.
```console
python setup.py build_ext --inplace
```

Install InfluxDB and Grafana. For windows, this helps:  
http://richardn.ca/2019/01/04/installing-influxdb-on-windows/  
Turn the slashes or use \\  
https://docs.influxdata.com/influxdb/v1.7/introduction/getting-started/  
in influx install directory use cmd or add to path:  
```console
influx
create database trade
```  
Grafana has an installer and connects easily to influxDb. Try getting data with a scriped SQL stmt.

- execute:
```console
python main train
``

Notes:  
- Tested with gpu compiled xgboost and lightgbm. should switch
automatically to using CPU only for pypi installed packages

## Running RL trainer gen_model_exit
- create entry models with gen_model.py which uploads predictions to db
    and creates an 'ex ... ' folder in models
- specifiy that folder name in config/gen_model_exit/<config.py>.entry_ex
- set the dates and min_entry_p_long accordingly to fetch entry preds and actually have entry
- run rl_start.py