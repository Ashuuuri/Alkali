# nfchain sampling results

## 20mpps  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 38 | 4 | 26 | 6+6+6+8 |
| 2 | 39 | 3 | 25 | 6+6+13 |
| 3 | 38 | 4 | 27 | 6+4+6+11 |
| 4 | 39 | 4 | 26 | 11+6+6+3 |
| 5 | 39 | 4 | 27 | 11+6+4+6 |

**Summary:** latency min=38 max=39 median=39; CU min=25 max=27 median=26

## Cross-workload summary

| Workload | target | Latency (median) | CU (median) | Stages (median) |
|----------|--------|-----------------|-------------|-----------------|
| 20mpps | 40 | 39 | 26 | 4 |
