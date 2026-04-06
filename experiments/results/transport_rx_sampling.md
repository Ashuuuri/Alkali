# transport_rx sampling results

## default  (target=100)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 69 | 1 | 3 | 3 |
| 2 | 69 | 1 | 3 | 3 |
| 3 | 69 | 1 | 3 | 3 |

**Summary:** latency min=69 max=69 median=69; CU min=3 max=3 median=3

## 20mpps  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 34 | 1 | 6 | 6 |
| 2 | 34 | 1 | 6 | 6 |
| 3 | 34 | 1 | 6 | 6 |

**Summary:** latency min=34 max=34 median=34; CU min=6 max=6 median=6

## 40mpps  (target=20)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 18 | 1 | 11 | 11 |
| 2 | 18 | 1 | 11 | 11 |
| 3 | 18 | 1 | 11 | 11 |

**Summary:** latency min=18 max=18 median=18; CU min=11 max=11 median=11

## Cross-workload summary

| Workload | target | Latency (median) | CU (median) | Stages (median) |
|----------|--------|-----------------|-------------|-----------------|
| default | 100 | 69 | 3 | 1 |
| 20mpps | 40 | 34 | 6 | 1 |
| 40mpps | 20 | 18 | 11 | 1 |
