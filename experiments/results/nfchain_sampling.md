# nfchain sampling results

## 20mpps_64B  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 38 | 4 | 29 | 9+8+6+6 |
| 2 | 37 | 4 | 29 | 9+8+6+6 |
| 3 | 38 | 4 | 28 | 6+8+6+8 |

**Summary:** latency min=37 max=38 median=38; CU min=28 max=29 median=29

## 20mpps_hot50  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 38 | 4 | 20 | 7+4+5+4 |
| 2 | 38 | 4 | 20 | 7+4+5+4 |
| 3 | 38 | 4 | 19 | 4+4+6+5 |

**Summary:** latency min=38 max=38 median=38; CU min=19 max=20 median=20

## 20mpps_hot90  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 35 | 4 | 11 | 4+3+2+2 |
| 2 | 38 | 4 | 10 | 3+3+2+2 |
| 3 | 38 | 4 | 11 | 3+3+2+3 |

**Summary:** latency min=35 max=38 median=38; CU min=10 max=11 median=11

## Cross-workload summary

| Workload | target | Latency (median) | CU (median) | Stages (median) |
|----------|--------|-----------------|-------------|-----------------|
| 20mpps_64B | 40 | 38 | 29 | 4 |
| 20mpps_hot50 | 40 | 38 | 20 | 4 |
| 20mpps_hot90 | 40 | 38 | 11 | 4 |
