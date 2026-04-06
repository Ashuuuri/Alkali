# nfchain sampling results

## 20mpps_64B  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 39 | 4 | 29 | 11+8+4+6 |
| 2 | 39 | 4 | 29 | 4+6+8+11 |
| 3 | 39 | 4 | 29 | 11+8+4+6 |

**Summary:** latency min=39 max=39 median=39; CU min=29 max=29 median=29

## 20mpps_4flows  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 81 | 4 | 16 | 4+4+4+4 |
| 2 | 76 | 4 | 16 | 4+4+4+4 |
| 3 | 103 | 4 | 16 | 4+4+4+4 |

**Summary:** latency min=76 max=103 median=81; CU min=16 max=16 median=16

## 20mpps_16flows  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 38 | 4 | 28 | 6+8+6+8 |
| 2 | 38 | 4 | 28 | 6+8+6+8 |
| 3 | 39 | 4 | 29 | 9+8+6+6 |

**Summary:** latency min=38 max=39 median=38; CU min=28 max=29 median=28

## 20mpps_64flows  (target=40)

| Run | Latency | Stages | CU total | Distribution |
|-----|---------|--------|----------|--------------|
| 1 | 39 | 4 | 29 | 9+8+6+6 |
| 2 | 38 | 4 | 28 | 6+8+6+8 |
| 3 | 39 | 4 | 29 | 9+8+6+6 |

**Summary:** latency min=38 max=39 median=39; CU min=28 max=29 median=29

## Cross-workload summary

| Workload | target | Latency (median) | CU (median) | Stages (median) |
|----------|--------|-----------------|-------------|-----------------|
| 20mpps_64B | 40 | 39 | 29 | 4 |
| 20mpps_4flows | 40 | 81 | 16 | 4 |
| 20mpps_16flows | 40 | 38 | 28 | 4 |
| 20mpps_64flows | 40 | 39 | 29 | 4 |
