# Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach

This codebase contains the python scripts for STHAN-SR, the model for the AAAI 2021 paper [link](https://ojs.aaai.org/index.php/AAAI/article/view/16127).

## Environment & Installation Steps
Python 3.6, Pytorch, Pytorch-Geometric and networkx.


```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset and follow preprocessing steps from [here](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking). 


## Run

Execute the following python command to train STHAN-SR: 
```python
python train_nyse.py -m NYSE -l 16 -u 64 -a 1 -e NYSE_rank_lstm_seq-8_unit-32_0.csv.npy 
python train_tse.py
python train_nasdaq.py -l 16 -u 64 -a 0.1
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{sawhney2021stock,
  title={Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach},
  author={Sawhney, Ramit and Agarwal, Shivam and Wadhwa, Arnav and Derr, Tyler and Shah, Rajiv Ratn},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={1},
  pages={497--504},
  year={2021}
}
```

