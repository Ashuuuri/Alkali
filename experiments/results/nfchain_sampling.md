# nfchain sampling results

## default  (target=100)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 77 | 4 | 13 | 3+3+3+4 |
| 2 | 81 | 4 | 13 | 4+3+3+3 |

**Summary:** latency min=77 max=81 median=79.0; CU min=13 max=13 median=13.0

## 20mpps  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 37 | 4 | 29 | 9+8+6+6 |
| 2 | 39 | 4 | 28 | 6+8+6+8 |

**Summary:** latency min=37 max=39 median=38.0; CU min=28 max=29 median=28.5

## Cross-workload summary

| Workload | target | Latency (median) | CU (median) | Stages (median) |
|----------|--------|-----------------|-------------|-----------------|
| default | 100 | 79.0 | 13.0 | 4.0 |
| 20mpps | 40 | 38.0 | 28.5 | 4.0 |
