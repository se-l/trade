# README #

Test model generation with default parameters and data in repo.  
This repo is being migrated to Julia.

### What is this repository for? ###

* Generate trading strategies
* Backtest strategies
* Evaluate strategy performance

### How do I get set up? ###

#### Windows
```console
git clone https://github.com/se-l/trade
cd trade
pip install -r requirements.txt
```

Installing TA-Lib might fail. Use
```console
conda install -c conda-forge ta-lib
```

2022 training in ./layers/predictions
2019 training in ./trader/train